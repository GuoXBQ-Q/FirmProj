package firmproj.base;
import com.kitfox.svg.A;
import firmproj.graph.CallGraph;
import firmproj.graph.CallGraphNode;
import firmproj.objectSim.*;
import firmproj.utility.LLMQuery;
import firmproj.utility.QueryJson;
import org.apache.logging.log4j.LogManager;
import org.apache.logging.log4j.Logger;
import soot.*;
import soot.Local;
import soot.SootMethod;
import soot.Unit;
import soot.Value;
import soot.jimple.*;

import java.sql.Array;
import java.util.*;

public class MethodLocal {
    private static final Logger LOGGER = LogManager.getLogger(MethodLocal.class);
    private final SootMethod sootMethod;

    public final static HashMap<SootMethod, HashMap<String, HashMap<Integer, List<String>>>> FindResult = new HashMap<>();

    private final HashMap<Value, AbstractClz> LocalToClz = new HashMap<>();
    private final HashMap<Value, List<String>> LocalToString = new HashMap<>();
    private final HashMap<Value, List<Integer>> LocalFromParams = new HashMap<>();

    private final HashMap<Value, MethodParamInvoke> LocalFromParamInvoke = new HashMap<>();
    private final HashMap<Value, List<List<String>>> LocalArray = new HashMap<>();

    private ValueContext ParentValueContext;
    private HashMap<Integer, List<String>> ParamsToString = new HashMap<>();

    private final List<String> returnValues = new ArrayList<>();

    private final HashMap<String, List<Integer>> InterestingInvoke = new HashMap<>();
    private final HashMap<String, HashMap<Integer, List<String>> > InterestingParamString = new HashMap<>();
    private Integer invokeStep = 0;
    private final Integer MAX_STEP = 4;
    private boolean isGetResult = false;

    //private List<String> unsolvedLocals = new ArrayList<>();

    public MethodLocal(SootMethod Method){
        this.sootMethod = Method;
    }

    //forward new
    public MethodLocal(SootMethod Method, ValueContext Values){
        this(Method);
        ParentValueContext = Values;
        if(ParentValueContext != null){
            InvokeExpr invokeExpr = (InvokeExpr) ParentValueContext.getCurrentUnit();
            if(invokeExpr != null){
                HashMap<Value,List<String>> currentValues =  ParentValueContext.getCurrentValues();
                Integer index = 0;
                HashMap<Integer, List<String>> ParamValue = new HashMap<>();
                for(Value param: invokeExpr.getArgs()){
                    if(currentValues.containsKey(param)){
                        ParamValue.put(index, currentValues.get(param));
                    }
                    else if(param instanceof Constant){
                        Object obj = SimulateUtil.getConstant(param);
                        if(obj != null) {
                            ParamValue.put(index, new ArrayList<>(List.of(param.toString())));
                        }
                    }
                    index++;
                }
                if (!ParamValue.isEmpty()) ParamsToString = ParamValue;
            }
        }
    }

    //backward new
    public MethodLocal(SootMethod Method, HashMap<String, List<Integer>> IntereInvoke, int step){
        this(Method);
        this.InterestingInvoke.putAll(IntereInvoke);
        invokeStep = step + 1;
    }

    public void doAnalysis(){
        Body body = null;
        try{
            body = this.sootMethod.retrieveActiveBody();
        }
        catch(Exception e){
            LOGGER.error(e);
        }
        if(body == null) return;
        if(invokeStep > MAX_STEP) return;
        if(MethodString.isStandardLibraryClass(sootMethod.getDeclaringClass())) return;
        try {
            if(sootMethod.getSignature().contains("com.liesheng.module_device.repository.DeviceService a()"))
                LOGGER.info("GOT");
            for (Unit unit : body.getUnits()) {
                Stmt stmt = (Stmt) unit;
                if (stmt instanceof AssignStmt) {
                    caseAssignStmt((AssignStmt) stmt);
                } else if (stmt instanceof IdentityStmt) {
                    caseIdentityStmt((IdentityStmt) stmt);
                } else if (stmt instanceof InvokeStmt) {
                    caseInvokeStmt((InvokeStmt) stmt);
                } else if (stmt instanceof ReturnStmt) {
                    caseReturnStmt((ReturnStmt) stmt);
                }
            }
        }catch (Exception e){LOGGER.error(e);}
    }

    private void caseReturnStmt(ReturnStmt stmt) {
        Value value = stmt.getOp();
        if(LocalToString.containsKey(value)) addValue(returnValues,LocalToString.get(value));
        else if(value instanceof Constant) {
            Object obj = SimulateUtil.getConstant(value);
            if(obj != null)
                returnValues.add(value.toString());
        }
        //TODO localTOClz
    }

    public void caseAssignStmt(AssignStmt stmt) {
        Value leftOp = stmt.getLeftOp();
        Value rightOp = stmt.getRightOp();

        if (leftOp instanceof Local) {
            if(LocalFromParams.containsKey(rightOp)) {
                List<Integer> keys = LocalFromParams.get(rightOp);
                LocalFromParams.put(leftOp, keys);
            }

            if(LocalFromParamInvoke.containsKey(rightOp)){
                LocalFromParamInvoke.put(leftOp, LocalFromParamInvoke.get(rightOp));
                LocalArray.remove(leftOp);
                LocalToString.remove(leftOp);
                LocalFromParams.remove(leftOp); //update the value
            }

            if(rightOp instanceof Local){
                if(LocalFromParams.containsKey(rightOp)){
                    LocalFromParams.put(leftOp, LocalFromParams.get(rightOp));
                }
                if(LocalToClz.containsKey(rightOp)){
                    LocalToClz.put(leftOp, LocalToClz.get(rightOp));
                }
                else if(LocalToString.containsKey(rightOp)){
                    addLocalTOString(leftOp, LocalToString.get(rightOp));
                }
            }
            if (rightOp instanceof NewExpr) {
                SootClass sootClz = ((NewExpr) rightOp).getBaseType().getSootClass();
                String clzName = ((NewExpr) rightOp).getBaseType().getClassName();
                if(MethodLocal.isCommonClz(clzName)) {
                    AbstractClz abstractClz = MethodLocal.CreateCommonClz(sootClz, sootMethod);
                    if(abstractClz != null)
                        LocalToClz.put(leftOp, abstractClz);
                    LocalFromParamInvoke.remove(leftOp);
                    LocalArray.remove(leftOp);
                    LocalToString.remove(leftOp);
                    LocalFromParams.remove(leftOp);
                }
            }
            else if(rightOp instanceof NewArrayExpr){
                NewArrayExpr arrayExpr = (NewArrayExpr) rightOp;
                Integer index = (Integer) SimulateUtil.getConstant(arrayExpr.getSize());
                if(index != null && index > 0 && index < 100){
                    List<List<String>> arrayList = new ArrayList<>();
                    int i = 0;
                    while(i < index) {
                        arrayList.add(new ArrayList<>());
                        i++;
                    }
                    LocalArray.put(leftOp, arrayList);
                }
                else{
                    LocalArray.remove(leftOp);
                }
            } else if (rightOp instanceof InvokeExpr) {
                HashMap<Value, List<String>> valueString = getInvokeExprValues((InvokeExpr) rightOp);
                ValueContext valueContext = new ValueContext(this.sootMethod,stmt,valueString);
                InvokeExpr invokeExpr = (InvokeExpr) rightOp;
                SootMethod invokeMethod = ((InvokeExpr) rightOp).getMethod();
                Value base = null;
                if(rightOp instanceof InstanceInvokeExpr){
                    base = ((InstanceInvokeExpr) rightOp).getBase();
                }
                if(MethodString.methodReturnParamInvoke.containsKey(invokeMethod) && !getInterestingInvoke().containsKey(invokeMethod.getSignature())){
                    MethodParamInvoke methodParamInvoke = MethodString.methodReturnParamInvoke.get(invokeMethod);
                    for(Integer param : clone(methodParamInvoke.param)){
                        Value arg = invokeExpr.getArg(param);
                        if(valueString.containsKey(arg)) {
                            HashMap<Integer, List<String>> tmp = new HashMap<>();
                            tmp.put(param, valueString.get(arg));
                            methodParamInvoke.addParamValue(tmp);
                        }
                        else if(arg instanceof Constant){
                            Object obj = SimulateUtil.getConstant(arg);
                            if(obj != null)
                                methodParamInvoke.addParamValue(  param, obj.toString());
                        }
                    }
                    LocalToString.remove(leftOp);
                    LocalArray.remove(leftOp);
                    LocalFromParamInvoke.remove(leftOp);
                    LocalToClz.remove(leftOp);
                    LocalFromParams.remove(leftOp);
                    addValue(LocalToString, leftOp, methodParamInvoke.toString());
                }
                else if(MethodString.methodToString.containsKey(invokeMethod)){
                    List<String> methodStr = MethodString.methodToString.get(invokeMethod);
                    LocalToString.put(leftOp, methodStr);

                    LocalArray.remove(leftOp);
                    LocalFromParamInvoke.remove(leftOp);
                    LocalToClz.remove(leftOp);
                    LocalFromParams.remove(leftOp);
                    return;
                }
                else if(MethodString.methodToFieldString.containsKey(invokeMethod)){
                    String fieldStr = MethodString.methodToFieldString.get(invokeMethod);
                    SootField field = Scene.v().getField(fieldStr);
                    if(MethodString.fieldToString.containsKey(fieldStr)){
                        LocalToString.put(leftOp, MethodString.fieldToString.get(fieldStr));
                    }
                    else if(MethodString.FieldSetByMethodInvoke.containsKey(field)){
                        List<MethodParamInvoke> methodParamInvokes = MethodString.FieldSetByMethodInvoke.get(field);
                        LocalToString.remove(leftOp);
                        for(MethodParamInvoke methodParamInvoke : methodParamInvokes){
                            addValue(LocalToString, leftOp, methodParamInvoke.toString());
                        }
                    }
                    LocalArray.remove(leftOp);
                    LocalFromParamInvoke.remove(leftOp);
                    LocalToClz.remove(leftOp);
                    LocalFromParams.remove(leftOp);
                    return;
                }
                if (base != null) {
                    InstanceInvokeExpr instanceInvokeExpr = (InstanceInvokeExpr) rightOp;

                    if(LocalToClz.containsKey(base)){
                        AbstractClz clz = LocalToClz.get(base);
                        clz.addValueContexts(valueContext);
                    }

                    if(MethodString.getMethodToString().containsKey(invokeMethod)) {
                        LocalToString.put(leftOp, MethodString.getMethodToString().get(invokeMethod));
                    }
                    else{
                        if(MethodString.classMaybeCache.containsKey(invokeMethod.getDeclaringClass())){
                            HashMap<String,List<String>> cache = MethodString.classMaybeCache.get(invokeMethod.getDeclaringClass());
                            if(instanceInvokeExpr.getArgs().size() ==1 && instanceInvokeExpr.getArg(0) instanceof Constant){
                                Object obj = SimulateUtil.getConstant(instanceInvokeExpr.getArg(0));
                                if(obj!=null) {
                                    String stringKey = obj.toString();
                                    if (cache.containsKey(stringKey)) {
                                        LocalToString.put(leftOp, cache.get(stringKey));
                                        //LOGGER.warn("[FIND CACHE VALUE] : {} - > {}", leftOp, cache.get(stringKey));
                                    }
                                }
                            }
                        }
                    }
                } else{
                    //TODO process normal method, like toJson, toString..; also the encrypt method.
                    HashMap<Integer, List<String>> paramIndexString = new HashMap<>();
                    for(int i=0;i<invokeExpr.getArgCount();i++){
                        Value arg = invokeExpr.getArg(i);
                        if(valueString.containsKey(arg)){
                            paramIndexString.put(i, valueString.get(arg));
                        }
                        else if(arg instanceof Constant){
                            Object obj = SimulateUtil.getConstant(arg);
                            if(obj != null)
                                paramIndexString.put(i, new ArrayList<>(List.of(obj.toString())));
                        }
                        else{
                            addValue(paramIndexString, i, "UNKNOWN: " + invokeMethod.getParameterType(i).toString());
                        }
                    }
                    List<String> result = getStaticInvokeReturn(invokeExpr, paramIndexString);
                    if(!result.isEmpty()){
                        LocalToString.put(leftOp, result);
                        LocalFromParamInvoke.remove(leftOp);
                        LocalFromParams.remove(leftOp);
                        LocalToClz.remove(leftOp);
                        LocalArray.remove(leftOp);
                    }
                    else{
                        MethodParamInvoke methodParamInvoke = new MethodParamInvoke(sootMethod, new ArrayList<>(), invokeMethod.getSignature() + MethodString.getContent(paramIndexString));
                        if(invokeMethod.getReturnType() instanceof ArrayType){
                            ArrayType arrayType = (ArrayType) invokeMethod.getReturnType();
                            if(arrayType.toString().toLowerCase().contains("byte")) return;
                            int LEN = 10;
                            List<List<String>> arrayList = new ArrayList<>();
                            List<String> arrayValue = new ArrayList<>();
                            if(!methodParamInvoke.invokeMethodSig.isEmpty())
                                arrayValue.add(MethodString.getContent(methodParamInvoke.invokeMethodSig));
                            else {
                                arrayValue.add(methodParamInvoke.toString());
                            }
                            int i = 0;
                            while(i < LEN) {
                                List<String> tmpValue = new ArrayList<>();
                                tmpValue.add("InvokeArray");
                                if(i == 0)
                                    tmpValue.add(MethodString.getContent(arrayValue));
                                tmpValue.add("get(" + i + ")");
                                arrayList.add(tmpValue);
                                i++;
                            }
                            LocalArray.put(leftOp, arrayList);
                            LocalFromParamInvoke.remove(leftOp);
                            LocalToString.remove(leftOp);
                            LocalToClz.remove(leftOp);
                            //LOGGER.info("297: New local Array: {}, {}, {},{}", stmt, leftOp, arrayList, index);
                        }
                        else{
                            LocalFromParamInvoke.put(leftOp, methodParamInvoke);
                            LocalArray.remove(leftOp);
                            LocalToString.remove(leftOp);
                            LocalToClz.remove(leftOp);
                            LocalFromParams.remove(leftOp);
                        }
                        if(sootMethod.getSignature().contains("getNewVersion"))
                            LOGGER.info("got");
                        if(isGetResult){
                            LLMQuery.generate(methodParamInvoke);
                        }
                    }
                }
            }
            else if(rightOp instanceof ArrayRef) {
                ArrayRef arrayRef = (ArrayRef) rightOp;
                Value arrayBase = arrayRef.getBase();
                LocalArray.remove(leftOp);
                LocalToString.remove(leftOp);
                LocalFromParamInvoke.remove(leftOp);
                LocalToClz.remove(leftOp);
                LocalFromParams.remove(leftOp);
                Object obj = SimulateUtil.getConstant(arrayRef.getIndex());
                if (obj != null) {
                    int arrayIndex = (Integer) obj;
                    if (LocalArray.containsKey(arrayBase)) {
                        try{
                            List<List<String>> array = LocalArray.get(arrayBase);
                            if(array.size() - 1 < arrayIndex) return;
                            List<String> arrayValue = array.get(arrayIndex);
                            if(arrayValue.isEmpty()) return;
                            LocalToString.put(leftOp, arrayValue);
                            LocalFromParams.remove(leftOp);
                            LocalArray.remove(leftOp);
                            LocalFromParamInvoke.remove(leftOp);//update the value
                        }
                        catch (Exception e){
                            //LOGGER.error("325: array index over, {},{},{},{}", clz, sootMethod, unit, localArray.get(arrayBase));
                        }
                    }
                }

            }
            else if (rightOp instanceof FieldRef) {
                SootField field = ((FieldRef) rightOp).getField();
                if(MethodString.fieldToString.containsKey(field.toString())) {
                    addValue(LocalToString, leftOp, MethodString.fieldToString.get((field.toString())));
                }
                else{
                    LocalToString.put(leftOp, new ArrayList<>(List.of(field.toString())));
                }
                LocalFromParamInvoke.remove(leftOp);
                LocalFromParams.remove(leftOp); //update the value
                LocalArray.remove(leftOp);
                LocalToClz.remove(leftOp);

            }
            else if (rightOp instanceof BinopExpr) {

            }
            else if(rightOp instanceof CastExpr) {
                Value op = ((CastExpr) rightOp).getOp();
                if(LocalToClz.containsKey(op))
                    LocalToClz.put(leftOp, LocalToClz.get(op));
                if(LocalFromParams.containsKey(op)){
                    if(!leftOp.equals(op))
                        LocalFromParams.remove(leftOp);
                    addValue(LocalFromParams, leftOp, LocalFromParams.get(op));
                }
            }
            else if(rightOp instanceof Constant){
                Object obj = SimulateUtil.getConstant(rightOp);
                if(obj != null) {
                    addValue(LocalToString,leftOp,obj.toString());
                    LocalFromParamInvoke.remove(leftOp);

                    LocalFromParams.remove(leftOp); //update the value
                    LocalToClz.remove(leftOp);

                }
            }
        }
        else if(leftOp instanceof ArrayRef){
            ArrayRef arrayRef = (ArrayRef) leftOp;
            Integer arrayIndex = (Integer) SimulateUtil.getConstant(arrayRef.getIndex());
            Value base = arrayRef.getBase();
            if(LocalArray.containsKey(base)) {
                if (arrayIndex != null) {
                    try {
                        List<String> arrayValue = LocalArray.get(base).get(arrayIndex);

                        if (rightOp instanceof Constant) {
                            Object obj = SimulateUtil.getConstant(rightOp);
                            if(obj != null)
                                addValue(arrayValue, obj.toString());
                        }
                        else if(LocalToString.containsKey(rightOp)){
                            addValue(arrayValue, LocalToString.get(rightOp));
                        }
                        else if(LocalToClz.containsKey(rightOp)){
                            AbstractClz abstractClz = LocalToClz.get(rightOp);
                            abstractClz.solve();
                            if(abstractClz.isSolved()) {
                                arrayValue.clear();
                                arrayValue.add(abstractClz.toString());
                            }
                            if(LocalFromParams.containsKey(rightOp)){
                                addValue(LocalFromParams, base, LocalFromParams.get(rightOp));
                            }
                        }
                        else if (LocalFromParamInvoke.containsKey(rightOp)){
                            MethodParamInvoke methodParamInvoke = new MethodParamInvoke(LocalFromParamInvoke.get(rightOp));
                            if(!methodParamInvoke.param.isEmpty())
                                addValue(LocalFromParams, base, methodParamInvoke.param);
                            arrayValue.clear();
                            arrayValue.addAll(new ArrayList<>(methodParamInvoke.invokeMethodSig));
                        }
                        else if (LocalFromParams.containsKey(rightOp)) {
                            arrayValue.clear();
                            arrayValue.addAll(MethodString.paramListToString(LocalFromParams.get(rightOp)));
                            addValue(LocalFromParams, base, LocalFromParams.get(rightOp));
                        }
                    }
                    catch (Exception e){
                        //LOGGER.error("348: array index over, {},{},{},{}", clz, sootMethod, unit, localArray.get(base));
                    }
                }
            }
            else {
                //TODO
            }
        }
        else {
            //LOGGER.warn(String.format("[%s] [SIMULATE][left unknown]: %s (%s)", this.hashCode(), stmt, leftOp.getClass()));
        }
    }

    public void caseInvokeStmt(InvokeStmt stmt) {
        String signature = stmt.getInvokeExpr().getMethod().toString();
        InvokeExpr invokeExpr = stmt.getInvokeExpr();
        Value base;
        if (invokeExpr instanceof InstanceInvokeExpr) {
            base = ((InstanceInvokeExpr) invokeExpr).getBase();
            HashMap<Value, List<String>> valueString = getInvokeExprValues(invokeExpr);
            ValueContext valueContext = new ValueContext(this.sootMethod, stmt, valueString);
            if(LocalToClz.containsKey(base)) LocalToClz.get(base).addValueContexts(valueContext);
        }
        else if( invokeExpr instanceof StaticInvokeExpr){
            //TODO process normal method, like toJson, toString..; also the encrypt method.
            if(invokeExpr.getMethod().getSignature().contains("checkNotNull"))
                return;
            HashMap<Value, List<String>> valueString = getInvokeExprValues(invokeExpr);
            ValueContext valuecontext = new ValueContext(this.sootMethod, stmt, valueString);
            //TODO localFromParam Transfer.

            //addChildValueContext(valuecontext);
            //MethodLocal nextMethodLocal = new MethodLocal(invokeExpr.getMethod(), valuecontext);
        }
    }

    public void caseIdentityStmt(IdentityStmt stmt){
        Value leftOperation = stmt.getLeftOp();
        Value rightOperation = stmt.getRightOp();
        if (leftOperation instanceof Local || leftOperation instanceof ArrayRef) {
            if (rightOperation instanceof ParameterRef) {
                ParameterRef ref = (ParameterRef) rightOperation;
                if(ParamsToString.containsKey(ref.getIndex())){
                    List<String> paramStr = ParamsToString.get(ref.getIndex());
                    addLocalTOString(leftOperation, paramStr);
                }
                else
                {
                    addValue(LocalFromParams,leftOperation, ref.getIndex());
                }
            }
        }
    }


    private void addLocalTOString(Value local, List<String> string){
        List<String> LocalStr = new ArrayList<>();
        if(LocalToString.containsKey(local)) LocalStr = LocalToString.get(local);
        if(LocalStr.isEmpty())
            LocalToString.put(local, string);
        else if(LocalStr.get(0).length() < string.get(0).length())
            LocalToString.put(local, string);
        else{
            //LOGGER.info("341: List: {}", string);
            addValue(LocalToString.get(local),string);
        }
    }

    public HashMap<Value, List<String>> getInvokeExprValues(InvokeExpr invokeExpr){
        HashMap<Integer, List<String>> interestingParamString = new HashMap<>();
        try {
            String invokeSIg = invokeExpr.getMethod().getSignature();
            //LOGGER.warn("[SIMULATE]: Get Values for Invoke: {}", invokeSIg);
            List<Integer> interestingParam = new ArrayList<>();

            boolean isInterestingInvoke = false;
            if (getInterestingInvoke().containsKey(invokeSIg)) {
                isInterestingInvoke = true;
                interestingParam = getInterestingInvoke().get(invokeSIg);
                //LOGGER.warn("[SIMULATE]: Find Interesting Invoke sig: {}=>{}", invokeSIg, interestingParam);
            }
            int i = 0;
            HashMap<String, List<Integer>> nextInvokeParam = new HashMap<>();
            HashMap<Integer, List<Integer>> invokeArgToMethodArg = new HashMap<>();
            HashMap<Value, List<String>> paramValue = new HashMap<>();

            List<CallGraphNode> callByNodes = CallGraph.findCaller(sootMethod.getSignature(), (sootMethod.isAbstract() || sootMethod.getDeclaringClass().isInterface()));
            callByNodes.removeIf(callGraphNode -> callGraphNode.getSootMethod().equals(sootMethod));
            for (Value value : invokeExpr.getArgs()) {
                if (value instanceof Constant) {
                    Object ob = SimulateUtil.getConstant(value);
                    if (isInterestingInvoke && interestingParam.contains(i)) {
                        if (ob != null) {
                            //LOGGER.warn("[SIMULATE]: Find Interesting Constant invoke value: {}=>{}", i, ob.toString());
                            addValue(interestingParamString, i, ob.toString());
                        } else {
                            addValue(interestingParamString, i, "null");
                        }
                    }
                } else if (value instanceof Local) {
                    if (LocalToClz.containsKey(value)) {
                        LocalToClz.get(value).solve();
                        String Clzstring = LocalToClz.get(value).getResult().toString();
                        if (LocalFromParams.containsKey(value) && !callByNodes.isEmpty()) {
                            LocalFromParamInvoke.put(value, new MethodParamInvoke(sootMethod, LocalFromParams.get(value), Clzstring));
                            addValue(nextInvokeParam, sootMethod.getSignature(), LocalFromParams.get(value));
                            invokeArgToMethodArg.put(i, clone(LocalFromParams.get(value)));
                        } else {
                            addValue(paramValue, value, Clzstring);
                            if (isInterestingInvoke && interestingParam.contains(i)) {
                                //LOGGER.warn("[SIMULATE]: Find Interesting Local invoke value: {}=>{}", i, paramValue.toString());
                                addValue(interestingParamString, i, paramValue.get(value));
                            }
                        }
                    }else if(LocalArray.containsKey(value)){
                        if(LocalFromParams.containsKey(value)){
                            addValue(nextInvokeParam, sootMethod.getSignature(), LocalFromParams.get(value));
                        }
                        addValue(paramValue, value, MethodString.getContent(LocalArray.get(value)));
                        if (isInterestingInvoke && interestingParam.contains(i)) {
                            //LOGGER.warn("[SIMULATE]: Find Interesting Local invoke value: {}=>{}", i, paramValue.toString());
                            addValue(interestingParamString, i, paramValue.get(value));
                        }
                    }
                    //LOGGER.warn("[SIMULATE]: Find Local Value for Invoke: {}", LocalToString.get(value));
                    else if (LocalFromParamInvoke.containsKey(value)) {
                        MethodParamInvoke methodParamInvoke = LocalFromParamInvoke.get(value);
                        List<Integer> params = methodParamInvoke.param;
                        if (methodParamInvoke.param.isEmpty() || methodParamInvoke.solved || callByNodes.isEmpty()) {
                            addValue(paramValue, value, methodParamInvoke.toString());
                            if (isInterestingInvoke && interestingParam.contains(i)) {
                                addValue(interestingParamString, i, paramValue.get(value));
                            }
                        } else{
                            addValue(nextInvokeParam, sootMethod.getSignature(), params);
                            invokeArgToMethodArg.put(i, clone(params));
                        }
                    } else if (LocalToString.containsKey(value)) {
                        //LOGGER.warn("[SIMULATE]: Find Interesting Solved Local Value: {}", LocalToString.get(value));
                        addValue(paramValue, value, LocalToString.get(value));
                        if (isInterestingInvoke && interestingParam.contains(i)) {
                            //LOGGER.warn("[SIMULATE]: Find Interesting Local invoke value: {}=>{}", i, MethodString.getContent(paramValue));
                            addValue(interestingParamString, i, paramValue.get(value));
                        }
                    } else if (LocalFromParams.containsKey(value)) {
                        //LOGGER.warn("[SIMULATE]: Find Local Value From Param: {}", LocalToString.get(value));
                        addValue(nextInvokeParam, sootMethod.getSignature(), LocalFromParams.get(value));
                        invokeArgToMethodArg.put(i, clone(LocalFromParams.get(value)));
                        if(callByNodes.isEmpty()){
                            addValue(paramValue, value, "UNKNOWN:" + invokeExpr.getArg(i).getType());
                            if (isInterestingInvoke && interestingParam.contains(i)) {
                                //LOGGER.warn("[SIMULATE]: CANT FIND invoke value: {}=>{}, {}", i, value, invokeExpr);
                                addValue(interestingParamString, i, paramValue.get(value));
                            }
                        }
                    } else {
                        addValue(paramValue, value, "UNKNOWN");
                        if (isInterestingInvoke && interestingParam.contains(i)) {
                            //LOGGER.warn("[SIMULATE]: CANT FIND invoke value: {}=>{}, {}", i, value, invokeExpr);
                            addValue(interestingParamString, i, paramValue.get(value));
                        }
                    }
                }
                i++;
            }
            if (!nextInvokeParam.isEmpty()) {
                if (callByNodes.isEmpty() || callByNodes.size() > 50) {
                    int index = 0;
                    for (Value arg : invokeExpr.getArgs()) {
                        if(arg instanceof Constant){
                            Object obj = SimulateUtil.getConstant(arg);
                            if (obj != null && isInterestingInvoke && interestingParam.contains(index)) {
                                //LOGGER.warn("[SIMULATE]: Find Interesting From invoke: {}=>{}, from: {}->{}", i, MethodString.getContent(paramValue), sootMethod, node.getSootMethod());
                                addValue(interestingParamString, index, obj.toString());
                            }
                            index++;
                            continue;
                        }
                        if(!paramValue.containsKey(arg) && (LocalFromParams.containsKey(arg) || LocalFromParamInvoke.containsKey(arg)))
                            addValue(paramValue, arg, "UNKNOWN:" + invokeExpr.getArg(index).getType());
                        if (isInterestingInvoke && interestingParam.contains(index) && paramValue.containsKey(arg)) {
                            //LOGGER.warn("[SIMULATE]: Find Interesting From invoke: {}=>{}, from: {}->{}", i, MethodString.getContent(paramValue), sootMethod, node.getSootMethod());
                            addValue(interestingParamString, index, paramValue.get(arg));
                        }
                        index++;
                    }

                } else {
                    getNodeCallToIfNotEq(nextInvokeParam, callByNodes);
                    if(callByNodes.size() > 20) callByNodes = callByNodes.subList(0,20);
                    for (CallGraphNode node : callByNodes) {
                        HashMap<String, List<Integer>> NextInvokeParam = clone(nextInvokeParam);
                        //LOGGER.warn("[SIMULATE]: Get Call By Nodes, {}==>{}", node.getSootMethod(), nextInvokeParam);
                        if (FindResult.containsKey(node.getSootMethod())) {
                            for(String str : NextInvokeParam.keySet()) {
                                HashMap<Integer, List<String>> result = FindResult.get(node.getSootMethod()).get(str);
                                if (result != null) {
                                    List<Integer> invokeIndexS = NextInvokeParam.get(str);
                                    invokeIndexS.removeAll(result.keySet());
                                }
                            }
                            NextInvokeParam.entrySet().removeIf(entry -> entry.getValue().isEmpty());
                        }

                        MethodLocal newMethodLocal = new MethodLocal(node.getSootMethod(), NextInvokeParam, this.invokeStep);
                        if (!NextInvokeParam.isEmpty()) {
                            if(this.isGetResult)
                                newMethodLocal.setGetResult(true);
                            newMethodLocal.doAnalysis();
                        }

                        if (!newMethodLocal.getInterestingParamString().isEmpty() || FindResult.containsKey(node.getSootMethod())) {
                            HashMap<Integer, List<String>> invokeResult = new HashMap<>();
                            if (!newMethodLocal.getInterestingParamString().isEmpty()) {
                                for(HashMap<Integer, List<String>> invokeResult1 : newMethodLocal.getInterestingParamString().values()){
                                    MethodString.addValue(invokeResult, invokeResult1);
                                }

                                if (!FindResult.containsKey(node.getSootMethod()))
                                    FindResult.put(node.getSootMethod(), new HashMap<>());

                                HashMap<String, HashMap<Integer, List<String>>> findResult = FindResult.get(node.getSootMethod());
                                for(String str : newMethodLocal.getInterestingParamString().keySet()){
                                    if(!findResult.containsKey(str))
                                        findResult.put(str, new HashMap<>());
                                    else if(!findResult.get(str).isEmpty()){
                                        MethodString.addValue(invokeResult, findResult.get(str));

                                    }
                                    if (!invokeResult.isEmpty())
                                        findResult.get(str).putAll(invokeResult); //update FindResult
                                }

                            } else {
                                for(String str : nextInvokeParam.keySet()){
                                    if(FindResult.get(node.getSootMethod()).containsKey(str))
                                        MethodString.addValue(invokeResult, FindResult.get(node.getSootMethod()).get(str));
                                }
                            }

                            for (MethodParamInvoke methodParamInvoke : LocalFromParamInvoke.values()) { //update the method param value of methodParamInvoke
                                HashMap<Integer, List<String>> tmpValues = new HashMap<>();
                                for(Map.Entry<Integer, List<String>> entry : invokeResult.entrySet()) {
                                    if (methodParamInvoke.param.contains(entry.getKey())) {
                                        addValue(tmpValues, entry.getKey(), entry.getValue());
                                    }
                                }
                                methodParamInvoke.addParamValue(tmpValues);
                                methodParamInvoke.solve();
                            }

                            for (Integer index : invokeResult.keySet()) {
                                for (Integer invokeArg : getInvokeArgFromMethodArg(invokeArgToMethodArg, index)) {
                                    Value value = invokeExpr.getArg(invokeArg);
                                    if (LocalFromParamInvoke.containsKey(value))
                                        addValue(paramValue, value, LocalFromParamInvoke.get(value).toString()); //if param value is methodParamInvoke
                                    else {
                                        addValue(paramValue, value, invokeResult.get(index));   //if param value if just local or other.
                                    }
                                    if (isInterestingInvoke && interestingParam.contains(invokeArg)) {
                                        //LOGGER.warn("[SIMULATE]: Find Interesting From invoke: {}=>{}, from: {}->{}", i, MethodString.getContent(paramValue), sootMethod, node.getSootMethod());
                                        addValue(interestingParamString, invokeArg, paramValue.get(value));
                                    }
                                }
                            }
                        }
                    }
                }
            }
            if(isInterestingInvoke && !interestingParamString.isEmpty()) {
                if (!getInterestingParamString().containsKey(invokeSIg))
                    getInterestingParamString().put(invokeSIg, new HashMap<>());
                getInterestingParamString().get(invokeSIg).putAll(interestingParamString);

            }
            return paramValue;//TODO valueString add
        }
        catch (Exception e){ LOGGER.error(e); }
        return new HashMap<>();
    }

    public static void getNodeCallToIfNotEq(HashMap<String, List<Integer>> nextInvokeParam, List<CallGraphNode> callByNodes){ //check if parent method or child method
        HashMap<String, List<Integer>> toAdd = new HashMap<>();
        for(String invokeSig : nextInvokeParam.keySet()) {
            for (CallGraphNode callGraphNode : callByNodes) {
                HashSet<CallGraphNode> callTo = callGraphNode.getCallTo();
                boolean hasCallTo = false;
                String pointerInvokeMethod = null;
                for (CallGraphNode callToNode : callTo) {
                    String sig = callToNode.getSootMethod().getSignature();
                    String subSig = callToNode.getSootMethod().getSubSignature();
                    if (invokeSig.equals(sig)) {
                        hasCallTo = true;
                        break;
                    }
                    else if(invokeSig.contains(subSig)){
                        pointerInvokeMethod = sig;
                    }

                }
                if(hasCallTo) continue;
                if(pointerInvokeMethod != null && !toAdd.containsKey(pointerInvokeMethod))
                    toAdd.put(pointerInvokeMethod, nextInvokeParam.get(invokeSig));
            }
        }

        nextInvokeParam.putAll(toAdd);
    }

    public List<Integer> getInvokeArgFromMethodArg(HashMap<Integer, List<Integer>> invokeArgToMethodArg, Integer methodArg){
        List<Integer> result = new ArrayList<>();
        for(Integer key : invokeArgToMethodArg.keySet()){
            List<Integer> value = invokeArgToMethodArg.get(key);
            if(value.contains(methodArg))
                addValue(result, key);
        }
        return result;

    }


    public static List<String> getStaticInvokeReturn(InvokeExpr invokeExpr, HashMap<Integer, List<String>> paramValues){
        SootMethod sootMethod = invokeExpr.getMethod();
        String sig = sootMethod.getSignature();
        SootClass sootClass = sootMethod.getDeclaringClass();
        List<String> result = new ArrayList<>();
        String value0 = "";
        if(paramValues.containsKey(0))
            value0 = MethodString.getContent(paramValues.get(0));
        else if(invokeExpr.getArg(0) instanceof Constant) {
            Object obj = SimulateUtil.getConstant(invokeExpr.getArg(0));
            if(obj != null)
                value0 = obj.toString();
        } else if(invokeExpr.getArgCount() !=0) {return result;}

        if(MethodString.classMaybeCache.containsKey(sootClass)){
            HashMap<String, List<String>> cache = MethodString.classMaybeCache.get(sootClass);
            if(cache.containsKey(value0))
                result.addAll(cache.get(value0));
            else{
                String str = sootMethod.getSignature() + '(' + value0 + ')';
                result.add(str);
            }
            return result;
        } else if(sig.contains("kotlin.TuplesKt: kotlin.Pair 'to'") || sig.contains("stringPlus") || sig.contains("java.net.URLEncoder: java.lang.String encode(java.lang.String,java.lang.String)")
        || sig.contains("format(java.lang.String,java.lang.Object[])")){
            String value1 = "";
            if(paramValues.containsKey(1))
                value1 = MethodString.getContent(paramValues.get(1));
            else if(invokeExpr.getArg(1) instanceof Constant) {
                Object obj = SimulateUtil.getConstant(invokeExpr.getArg(0));
                if (obj != null)
                    value1 = obj.toString();
            }
            else return result;

            if(sig.contains("kotlin.TuplesKt: kotlin.Pair 'to'"))
                result.add(value0 + '=' + value1);
            else if(sig.contains("stringPlus"))
                result.add(value0 + value1);
            else if(sig.contains("java.net.URLEncoder: java.lang.String encode"))
                result.add("encode(" + value0 + "," + value1 + ")");
            else if(sig.contains("format(java.lang.String,java.lang.Object[])"))
                result.add("format(0=" + value0 + ", 1=" + value1 + ")");
            return result;
        } else if (sig.contains("listOf") || sig.contains("valueOf") || sig.contains("Boxing") || sig.contains("mapOf") || sig.contains("mutableMapOf")) {
            result.add(value0);
            return result;
        } else if( sig.contains("currentTimeMillis")){
            result.add("(currentTimeMills)" + System.currentTimeMillis());
            return result;
        } else if(sig.contains("emptyList()")){
            return new ArrayList<>(List.of("null"));
            //TODO try to do methodlocal to get return.
        }

        return result;
    }

    public static <K, V> void addValue(Map<K, List<V>> map, K key, V value) {
        map.computeIfAbsent(key, k -> new ArrayList<>());
        List<V> values = map.get(key);
        if(!values.contains(value)){
            values.add(value);
        }
    }

    public static <V> void addValue(List<V> list,  V value) {
        if(!list.contains(value)){
            list.add(value);
        }
    }

    public static <V> void addValue(List<V> list1, List<V> list2){
        for(V value: list2){
            addValue(list1, value);
        }
    }

    public static <K, V> void addValue(Map<K, List<V>> map, K key, List<V> values) {
        map.computeIfAbsent(key, k -> new ArrayList<>());
        List<V> key_values = map.get(key);
        for(V value: values) {
            if (!key_values.contains(value)) {
                key_values.add(value);
            }
        }
    }

    public static boolean isCommonClz(String className){
        if(className.contains("Map") || className.equals("org.json.JSONObject") || className.contains("java.lang.StringBuilder") || className.contains("java.lang.StringBuffer") || className.contains("java.net.URL"))
            return true;
        return false;
    }

    public ValueContext getParentValueContext() {
        return ParentValueContext;
    }

    public List<String> getReturnValues() {
        return returnValues;
    }

    public HashMap<Integer, List<String>> getParamsToString() {
        return ParamsToString;
    }

    public HashMap<String, HashMap<Integer, List<String>> > getInterestingParamString() {
        return InterestingParamString;
    }

    public HashMap<String, List<Integer>> getInterestingInvoke() {
        return InterestingInvoke;
    }

    public static AbstractClz CreateCommonClz(SootClass clz, SootMethod currentMethod){
        if(clz.getName().contains("Map"))
            return new HashmapClz(clz, currentMethod);
        else if(clz.getName().contains("StringBuilder") || clz.getName().contains("StringBuffer"))
            return new StringClz(clz, currentMethod);
        else if (clz.getName().contains("java.net.URL")) {
            return new UrlClz(clz, currentMethod);
        } else if(!MethodString.isStandardLibraryClass(clz))
            return new CustomClz(clz, currentMethod);

        return null;
    }

    public void setGetResult(boolean getResult) {
        isGetResult = getResult;
    }

    public SootMethod getSootMethod(){
        return this.sootMethod;
    }

    public HashMap<Value, AbstractClz> getLocalToClz() {
        return LocalToClz;
    }

    public HashMap<Value, List<String>> getLocalToString() {
        return LocalToString;
    }

    public HashMap<Value, List<Integer>> getLocalFromParams() {
        return LocalFromParams;
    }

    public static <T> List<T> clone(List<T> ls) {
        return new ArrayList<T>(ls);
    }

    public static <K,V> HashMap<K, List<V>> clone(HashMap<K, List<V>> input){
        HashMap<K, List<V>> result = new HashMap<>();
        for (K key : input.keySet()) {
            result.put(key, new ArrayList<>(input.get(key)));
        }
        return result;
    }

//    public List<String> getUnsolvedLocals() {
//        return unsolvedLocals;
//    }
}
