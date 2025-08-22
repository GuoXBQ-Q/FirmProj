package firmproj.base;
import firmproj.graph.*;
import firmproj.objectSim.*;
import firmproj.utility.*;
import org.apache.logging.log4j.LogManager;
import org.apache.logging.log4j.Logger;
import org.jetbrains.annotations.NotNull;
import org.jf.dexlib2.iface.reference.MethodProtoReference;
import soot.*;
import soot.jimple.*;
import soot.tagkit.*;
import soot.util.Chain;

import java.util.*;
import java.util.regex.*;

public class MethodString {
    private static final Logger LOGGER = LogManager.getLogger(MethodString.class);

    public static final HashMap<SootMethod, String> methodToFieldString = new HashMap<>();

    public static final HashMap<String, List<String>> fieldToString = new HashMap<>();

    public static final HashMap<SootMethod, List<String>> methodToString = new HashMap<>();

    public static final HashMap<SootField, List<MethodParamInvoke>> FieldSetByMethodInvoke = new HashMap<>();// field set by Param Invoke

    public static final HashMap<String, List<SootMethod>> fieldSetByPointMethod = new HashMap<>();

    public static final HashMap<SootMethod, HashSet<SootField>> methodToInvokeSetFields = new HashMap<>();

    public static final HashMap<SootMethod, MethodParamInvoke> methodReturnParamInvoke = new HashMap<>();

    public static final HashMap<String, List<SootField>> classWithOuterInnerFields = new HashMap<>();

    public static final HashMap<SootClass, HashMap<String, List<String>>> classMaybeCache = new HashMap<>();

    public static final HashMap<SootClass, HashMap<SootMethod, HashMap<Integer, SootField>>> allClzWithSetFieldMethod = new HashMap<>();

    public static final List<String> findUrl = new ArrayList<>();

    public static void init(){
        Chain<SootClass> classes = Scene.v().getClasses();
        for (SootClass sootClass : classes) {
            String headString = sootClass.getName().split("\\.")[0];
            if (headString.contains("android") || headString.contains("kotlin") || headString.contains("java"))
                continue;
            //LOGGER.info(sootClass.getName().toString());
            if(isStandardLibraryClass(sootClass)) continue;
            for (SootMethod sootMethod : clone(sootClass.getMethods())) {
                if (!sootMethod.isConcrete())
                    continue;
                Body body = null;
                try {
                    body = sootMethod.retrieveActiveBody();
                } catch (Exception e) {
                    //LOGGER.error("Could not retrieved the active body {} because {}", sootMethod, e.getLocalizedMessage());
                }
                if (body == null)
                    continue;
                if (isCommonType(sootMethod.getReturnType()) && sootMethod.getDeclaringClass().isApplicationClass()) {
                    try {
                        unsolvedMethodString = new ArrayList<>();
                        //LOGGER.info("176: " + sootMethod.getReturnType() + ":" + sootMethod.getSignature());
                        GetMethodToString(sootMethod);
                    } catch (Throwable e) {
                        //LOGGER.error("error: " + sootMethod + ":" + e);
                    }
                }
            }
        }
        GetAllMethodSetField();
        LOGGER.info("Start retrieve field set by callback");
        RetrieveFieldSetByCallBack();
        LOGGER.info("Start Merge Method and Field String");
        MergeMethodAndFieldString();
        LOGGER.info("Start retrieve MethodParamInvoke");
        RetrieveAllMethodParamInvoke();
        LOGGER.info("Start replace All field String");
        ReplaceFieldString();
        LOGGER.info("Start find Url");
        findAllPossibleUrl();

        LOGGER.info("SIZE: {},{},{},{}", methodToFieldString.size() , fieldToString.size() ,allClzWithSetFieldMethod.size(),classWithOuterInnerFields);
        LOGGER.warn("MethodToField: {}\n", methodToFieldString);
        LOGGER.warn("MethodToString: {}\n",methodToString);
        LOGGER.warn("FieldToString: {}\n", fieldToString);
        LOGGER.warn("allClzWithSetFieldMethod: {}\n", allClzWithSetFieldMethod);
        LOGGER.warn("FieldSetByMethodInvoke: {}\n", FieldSetByMethodInvoke);
        LOGGER.warn("MethodToInvokeSetFields: {}\n", methodToInvokeSetFields);
        LOGGER.warn("classWithOuterInnerFields: {}\n", classWithOuterInnerFields);

        //LOGGER.warn("methodReturnParamInvoke: " + methodReturnParamInvoke);

    }
    private static List<String> unsolvedMethodString;
    private static final HashSet<String> visitedMethod = new HashSet<>();

    public static List<String> GetMethodToString(SootMethod sootMethod){
        List<String> mstrs = new ArrayList<>();
        List<Unit> VisitedUnits = new ArrayList<>();
        if(!sootMethod.getDeclaringClass().isApplicationClass() || isStandardLibraryClass(sootMethod.getDeclaringClass())) return mstrs;

        if(methodToString.containsKey(sootMethod)) return methodToString.get(sootMethod);

        if(visitedMethod.contains(sootMethod.getSignature())) return mstrs;
        visitedMethod.add(sootMethod.getSignature());

        Body body = null;
        try {
            body = sootMethod.retrieveActiveBody();
        } catch (Exception e) {
            //LOGGER.error("Could not retrieved the active body {} because {}", sootMethod, e.getLocalizedMessage());
        }
        if(body != null) {
            for (Unit unit : body.getUnits()) {
                //LOGGER.info("234: UNIT: " + unit.toString());
                if (unit instanceof ReturnStmt) {
                    ReturnStmt returnStmt = (ReturnStmt) unit;

                    Value retValue = returnStmt.getOp();
                    if(retValue==null) continue;
                    if (retValue instanceof Constant) {
                        Object obj = SimulateUtil.getConstant(retValue);
                        if(obj != null) {
                            String str = obj.toString();
                            if (str.equals("\"\"") || str.equals("null") || str.equals("0") || str.isEmpty()) continue;
                            addValue(methodToString, sootMethod, str);
                            TryAddStringToInterface(sootMethod);
                            LOGGER.info("266:New Method To String: " + sootMethod.getSignature() + ":" + str + ";" + getContent(methodToString.get(sootMethod)));
                            mstrs.add(str);
                        }
                        //return mstrs;
                    }
                    else if (retValue instanceof Local) {
                        Local local = (Local) retValue;
                        for (Unit u : body.getUnits()) {
                            if(VisitedUnits.contains(u)) continue;
                            //LOGGER.info("254: UNIT: " + u.toString());
                            VisitedUnits.add(u);
                            if(u == unit) break;
                            if (u instanceof AssignStmt) {
                                AssignStmt assignStmt = (AssignStmt) u;
                                if (assignStmt.getLeftOp().equals(local)) {
                                    Value rightOp = assignStmt.getRightOp();
                                    if (rightOp instanceof FieldRef) {
                                        FieldRef fieldRef = (FieldRef) rightOp;
                                        if (fieldRef.getField() == null || fieldRef.getField().getDeclaringClass() == null) {
                                            continue;
                                        }
                                        if (fieldRef.getField().getDeclaringClass().isApplicationClass()) {
                                            SootField field = fieldRef.getField();
                                            if (!methodToFieldString.containsKey(sootMethod)) {
                                                methodToFieldString.put(sootMethod, field.toString());
                                                //LOGGER.info("New Method ret field: " + sootMethod.getSignature() + ":" + field);
                                                if (fieldToString.containsKey(field.toString())) {
                                                    List<String> strs = fieldToString.get(field.toString());
                                                    addValue(mstrs, strs);
                                                    //LOGGER.info("238:New Method To String: " + sootMethod.getSignature() + ":" + strs + ";" + MethodString.getContent(methodToString.get(sootMethod)));
                                                    //return mstrs;
                                                }
                                            }
                                        }
                                    } else if (rightOp instanceof InvokeExpr) {
                                        List<String> strs = new ArrayList<>();
                                        SootMethod nextMethod = ((InvokeExpr) rightOp).getMethod();
                                        if(isCommonType(nextMethod.getReturnType())) {
                                            if (!nextMethod.toString().equals(sootMethod.toString())) { // Is Recursion function
                                                unsolvedMethodString.add(sootMethod.getSignature());
                                                if(unsolvedMethodString.contains(nextMethod.getSignature())) return mstrs;
                                                strs = GetMethodToString(nextMethod);
                                            }
                                        }
                                        if (!strs.isEmpty()) {
                                            addValue(mstrs, strs);
                                            //LOGGER.info("255:New Method To String: " + sootMethod.getSignature() + ":" + mstrs + ";" + MethodString.getContent(methodToString.get(sootMethod)));
                                        }
                                        else{
                                            mstrs.clear();
                                            methodToFieldString.remove(sootMethod);
                                        }
                                        //return mstrs;
                                    }
                                    else if(rightOp instanceof Constant){
                                        Object obj = SimulateUtil.getConstant(rightOp);
                                        if(obj!=null) {
                                            mstrs.add(rightOp.toString());
                                            //LOGGER.info("314:New Method To String: " + sootMethod.getSignature() + ":" + mstrs + ";" + MethodString.getContent(methodToString.get(sootMethod)));
                                        }
                                    }
                                    else if(rightOp instanceof BinopExpr){
                                        //TODO
                                    }
                                }
                            }
                        }
                    }

                }
            }
            if(!mstrs.isEmpty()) {
                addValue(methodToString, sootMethod, mstrs);
                TryAddStringToInterface(sootMethod);
            }
        }
        return mstrs;
    }

    public static void GetAllMethodSetField(){
        //TODO
        visitedMethod.clear();
        Chain<SootClass> classes = Scene.v().getClasses();
        for (SootClass sootClass : classes) {
            HashMap<SootMethod, HashMap<Integer, SootField>> clzResult;
            if(isStandardLibraryClass(sootClass)) continue;
            clzResult = GetMethodSetField(sootClass);
            if(!clzResult.isEmpty()){
                //LOGGER.info("GetMethodSetField: {}--{}", sootClass, clzResult);
            }
        }
    }

    public static HashMap<SootMethod, HashMap<Integer, SootField>> GetMethodSetField(SootClass clz){
        String headString = clz.getName().split("\\.")[0];
        if (headString.contains("android") || headString.contains("kotlin") || headString.contains("java") || headString.contains("thingClips"))
            return new HashMap<>();

        if(!allClzWithSetFieldMethod.containsKey(clz))
            allClzWithSetFieldMethod.put(clz, new HashMap<>());
        else{
            return allClzWithSetFieldMethod.get(clz);
        }
        HashMap<SootMethod, HashMap<Integer, SootField>> result = allClzWithSetFieldMethod.get(clz);

        processStaticField(clz);
        for(SootMethod sootMethod : clz.getMethods()){
            HashMap<Integer, SootField> fieldWithParam = GetSetField(sootMethod);
            if(!fieldWithParam.isEmpty()) {
                result.put(sootMethod, fieldWithParam);
                //LOGGER.info("220: GetSetField: {}--{}--{}",clz, sootMethod, fieldWithParam);
            }
            else result.remove(sootMethod);

        }
        return result;
    }

    public static void TryAddStringToInterface(SootMethod method){
        for(SootClass interfaceClz: method.getDeclaringClass().getInterfaces()){
            try{
                SootMethod interfaceMethod = interfaceClz.getMethodByName(method.getName());
                addValue(methodToString, interfaceMethod, methodToString.get(method));
                LOGGER.info("222: Add String To Interface: {} -> {} - {}", methodToString.get(method), interfaceMethod.getSignature(), method.getSignature());
            }
            catch (Exception e){LOGGER.error(e);}
        }
    }

    public static void processStaticField(SootClass sc){
        for (SootField field : sc.getFields()) {
            if (field.isStatic()) {
                for (Tag tag : field.getTags()) {
                    if (tag instanceof ConstantValueTag) {
                        ConstantValueTag constantValueTag = (ConstantValueTag) tag;
                        Object obj = SimulateUtil.getConstant(constantValueTag.getConstant());
                        if(obj != null) {
                            String str = obj.toString();
                            addValue(fieldToString, field.toString(), str);
                        }
                    }
                }
            }
        }
    }

    public static void RetrieveFieldSetByCallBack(){
        for(SootField field : FieldSetByMethodInvoke.keySet()){
            List<MethodParamInvoke> methodParamInvokes = FieldSetByMethodInvoke.get(field);
            for(MethodParamInvoke methodParamInvoke : methodParamInvokes){
                SootMethod fromMethod = methodParamInvoke.sootMethod;
                if(CallGraph.callBackClassesWithPoint.containsKey(fromMethod.getDeclaringClass())){
                    CallBackPoint callBackPoint = CallGraph.callBackClassesWithPoint.get(fromMethod.getDeclaringClass());
                    addValue(callBackPoint.relatedFields, field.toString());
                    List<SootMethod> points = callBackPoint.relatedPoint;
                    if(!points.isEmpty()) {
                        fieldSetByPointMethod.put(field.toString(), points);
                        String newFieldStr = getFieldPointString(field.toString(), points);
                        addValue(fieldToString, field.toString(), newFieldStr);
                        LOGGER.debug("294: NEW FIELD TO POINT: {} -> {}", field, newFieldStr);
                    }
                }
            }
        }
    }

    public static void RetrieveAllMethodParamInvoke(){
        HashSet<MethodParamInvoke> visited = new HashSet<>();
        HashSet<MethodParamInvoke> allMethodParamInvoke = new HashSet<>();
        for(List<MethodParamInvoke> methodParamInvokes : FieldSetByMethodInvoke.values()){
            allMethodParamInvoke.addAll(methodParamInvokes);
        }
        for(MethodParamInvoke methodParamInvoke : methodReturnParamInvoke.values()){
            Type type = methodParamInvoke.sootMethod.getReturnType();
            if(isCommonType(type))
                allMethodParamInvoke.add(methodParamInvoke);
        }

        for(MethodParamInvoke methodParamInvoke : allMethodParamInvoke){
            if(methodParamInvoke.sootMethod.getSignature().equals("<com.yamaha.hs.hpctrl.firmware.FirmwareServerInfo: java.net.URL url(java.lang.String)>"))
                LOGGER.info("got");
            LOGGER.info("No.{} of total: {} Now: {}, ", visited.size(), allMethodParamInvoke.size(), methodParamInvoke.sootMethod.getSignature());
            if(visited.contains(methodParamInvoke)) continue;
            visited.add(methodParamInvoke);
            HashMap<String, List<Integer>> nextInvokeParam = new HashMap<>();
            HashMap<Integer, List<String>> paramValue = new HashMap<>();
            SootMethod sootMethod = methodParamInvoke.sootMethod;
            String sig = sootMethod.getSignature().toLowerCase();
            if(isStandardLibraryClass(sootMethod.getDeclaringClass()) || sig.contains(".internal.")|| sig.contains("tostring()") || sig.contains("format(") || sig.contains("lang.reflect")) continue;
            nextInvokeParam.put(sootMethod.getSignature(), methodParamInvoke.param);

            if(methodParamInvoke.invokeMethodSig.toString().length() > 5000) continue;

            List<CallGraphNode> callByNodes = CallGraph.findCaller(sootMethod.getSignature(), (sootMethod.isAbstract() || sootMethod.getDeclaringClass().isInterface()));
            if(callByNodes.size() > 20) continue;
            //if(sootMethod.getSignature().equals("<com.yamaha.sc.hpcontroller.firmwareDownload.FirmwareServerInfo: java.net.URL url(java.lang.String)>"))
                //LOGGER.error("got : {}", callByNodes.toString());
            if(callByNodes.isEmpty()){
                for(Integer arg : methodParamInvoke.param){
                    addValue(paramValue, arg, "UNKNOWN <"+ sootMethod.getSubSignature() + ">[$" + arg + "]");
                }
            }
            else {
                MethodLocal.getNodeCallToIfNotEq(nextInvokeParam, callByNodes);
                int timeout = 10;
                TimeoutTaskExecutor timeoutTaskExecutor = new TimeoutTaskExecutor(timeout);
                timeoutTaskExecutor.executeWithTimeout(() -> {
                    for (CallGraphNode node : callByNodes) {
                        HashMap<String, List<Integer>> Next = clone(nextInvokeParam);
                        //LOGGER.warn("[SIMULATE]: Get Call By Nodes, {}==>{}", node.getSootMethod(), nextInvokeParam);
                        MethodLocal newMethodLocal = new MethodLocal(node.getSootMethod(), Next, 1);
                        newMethodLocal.doAnalysis();
                        if (newMethodLocal.getInterestingParamString().containsKey(sootMethod.getSignature())) {
                            HashMap<Integer, List<String>> invokeResult = newMethodLocal.getInterestingParamString().get(sootMethod.getSignature());
                            if(!invokeResult.isEmpty()) {
                                addValue(paramValue, invokeResult);
                                if(methodParamInvoke.solved && !methodReturnParamInvoke.containsValue(methodParamInvoke)){
                                    LLMQuery.generate(methodParamInvoke);
                                }
                            }
                        }
                    }
                });
                timeoutTaskExecutor.shutdown();
                if(paramValue.isEmpty()) {
                    for (Integer arg : methodParamInvoke.param) {
                        addValue(paramValue, arg, "UNKNOWN <" + sootMethod.getSubSignature() + ">[$" + arg + "]");
                    }
                }
            }
            methodParamInvoke.addParamValue(paramValue);
            methodParamInvoke.solve();
            methodParamInvoke.param.clear();
        }
    }

    private static @NotNull String getFieldPointString(String fieldStr, List<SootMethod> points) {
        String newFieldStr;
        StringBuilder stringBuilder = new StringBuilder(fieldStr + "->SetByPoint:[");
        for(SootMethod sootMethod : points){
            stringBuilder.append(sootMethod.getName());
            stringBuilder.append(",");
        }
        stringBuilder.deleteCharAt(stringBuilder.length()-1);
        stringBuilder.append("]");
        newFieldStr = stringBuilder.toString();
        return newFieldStr;
    }

    public static void MergeMethodAndFieldString(){
        for(SootMethod method : methodToFieldString.keySet()){
            String fieldStr = methodToFieldString.get(method);
            if(fieldToString.containsKey(fieldStr)) {
                addValue(methodToString, method, fieldToString.get(fieldStr));
            }
        }
    }

    public static void ReplaceFieldString(){
        HashSet<MethodParamInvoke> visited = new HashSet<>();
        HashSet<MethodParamInvoke> allMethodParamInvoke = new HashSet<>();
        for(List<MethodParamInvoke> methodParamInvokes : FieldSetByMethodInvoke.values()){
            allMethodParamInvoke.addAll(methodParamInvokes);
        }
        allMethodParamInvoke.addAll(methodReturnParamInvoke.values());

        HashSet<List<String>> visitedList = new HashSet<>();
        for(List<String> fieldStrS : fieldToString.values()){
            List<String> cloneList = clone(fieldStrS);
            if(visitedList.contains(fieldStrS)) continue;
            visitedList.add(fieldStrS);

            fieldStrS.removeAll(List.of(""));
            int index = 0;
            for(String fieldStr : fieldStrS) {
                visitedStr.clear();
                visitedStr.add(fieldStrS.toString());
                String tempResult = retrieveAllfieldString(fieldStr, fieldStr);
                if(!tempResult.equals(fieldStr) && tempResult.length() < fieldStr.length() + 5000)
                    cloneList.set(index, tempResult);
                else if(fieldSetByPointMethod.containsKey(fieldStr))
                    cloneList.set(index, getFieldPointString(fieldStr, fieldSetByPointMethod.get(fieldStr)));
                index++;
            }
            fieldStrS.clear();
            fieldStrS.addAll(cloneList);
            removeDuplicates(fieldStrS);
        }

        for(String fieldStr : fieldToString.keySet()){
            SootField field = Scene.v().getField(fieldStr);
            String fieldContent = getContent(fieldToString.get(fieldStr));
            Type type = field.getType();
            if(type.toString().toLowerCase().contains("url")){
                if(isFirmRelatedUrl(fieldContent))
                    findUrl.add(field.getName() + "=" + fieldContent);
            }
            else if(type.toString().contains("String") ){
                if(field.getName().toLowerCase().contains("url") && fieldContent.contains("://") && fieldContent.contains("http"))
                    findUrl.add(field.getName() + "=" + fieldContent);
                else if(isFirmRelatedUrl(fieldContent))
                    findUrl.add(field.getName() + "=" + fieldContent);
            }
        }

        for(MethodParamInvoke methodParamInvoke : allMethodParamInvoke){
            if(visited.contains(methodParamInvoke)) continue;
            visited.add(methodParamInvoke);
            List<String> invokes = methodParamInvoke.invokeMethodSig;
            if(invokes.isEmpty()) continue;
            int index = 0;
            for(String invoke : invokes) {
                if(invoke.isEmpty()) continue;
                visitedStr.clear();
                String tmpResult = retrieveAllfieldString(invoke, invoke);
                if(!tmpResult.equals(invoke)  && tmpResult.length() < invoke.length() + 5000){
                    invokes.set(index, tmpResult);
                }
                index++;
            }
            removeDuplicates(invokes);
        }

        for(List<String> methodStrS : methodToString.values()){
            methodStrS.removeAll(List.of(""));
            int index = 0;
            for(String methodStr : methodStrS) {
                visitedStr.clear();
                String tempResult = retrieveAllfieldString(methodStr, methodStr);
                if(!tempResult.equals(methodStr)  && tempResult.length() < methodStr.length() + 5000)
                    methodStrS.set(index, tempResult);
                index++;
            }
            removeDuplicates(methodStrS);
        }
    }

    private static final HashSet<String> visitedStr = new HashSet<>();

    public static String retrieveAllfieldString(String startField, String str){
        if(visitedStr.contains(startField)) return startField;
        String tmpResult = str;
        HashSet<String> fields = FirmwareRelated.matchFields(str);
        if(fields.isEmpty()) return tmpResult;
        for(String field : fields){
            if(!visitedStr.contains(field)) {
                if(isStandardLibraryField(field)) continue;
                if (fieldToString.containsKey(field)) {
                    String newFieldStr = null;
                    List<String> fieldStr = fieldToString.get(field);
                    newFieldStr = getContent(fieldStr);

                    if (fieldStr.toString().contains(startField) || visitedStr.contains(fieldStr.toString())) {
                        //fieldStr.replaceAll(s -> s.replace(field, ""));
                        return tmpResult; //ring
                    }

                    fieldStr.replaceAll(s -> s.replace(field, ""));
                    visitedStr.add(startField);
                    String resultStr = retrieveAllfieldString(field, newFieldStr);
                    if (tmpResult.contains(resultStr))
                        tmpResult = tmpResult.replace(field, "");
                    else if(resultStr.length() < field.length() + 5000)
                        tmpResult = tmpResult.replace(field, resultStr);
                }
            }

        }
        return tmpResult;
    }

    public static void findAllPossibleUrl(){
        HashSet<MethodParamInvoke> visited = new HashSet<>();
        HashSet<MethodParamInvoke> allMethodParamInvoke = new HashSet<>();
        for(List<MethodParamInvoke> methodParamInvokes : FieldSetByMethodInvoke.values()){
            allMethodParamInvoke.addAll(methodParamInvokes);
        }
        allMethodParamInvoke.addAll(methodReturnParamInvoke.values());

        for(MethodParamInvoke methodParamInvoke : allMethodParamInvoke){
            if(visited.contains(methodParamInvoke)) continue;
            visited.add(methodParamInvoke);
            String returnType = methodParamInvoke.sootMethod.getReturnType().toString().toLowerCase();
            if(returnType.contains("string")){
                String str = getContent(methodParamInvoke.invokeMethodSig);
                if(isFirmRelatedUrl(str) && str.length() < 1000) {
                    LOGGER.debug("FIND URL: {} ,from String method {} ", str, methodParamInvoke.sootMethod);
                    findUrl.add(str);
                }
            }
            if(returnType.contains("url") && isFirmRelatedUrl(methodParamInvoke.invokeMethodSig.toString())){
                if(methodParamInvoke.solved) {
                    LOGGER.debug("FIND URL : {} ,from Url method {}", methodParamInvoke.toString(), methodParamInvoke.sootMethod);
                    findUrl.add(methodParamInvoke.toString());
                }
            }
        }
    }


    public static HashMap<Integer, SootField> GetSetField(SootMethod sootMethod){

        try {
            if (sootMethod.getSignature().contains("AM_serverZone_URL(java.lang.String,int)"))
                LOGGER.info("TARGET");
            SootClass clz = sootMethod.getDeclaringClass();
            HashMap<SootMethod, HashMap<Integer, SootField>> clzMethod = allClzWithSetFieldMethod.get(clz);

            if (!clzMethod.containsKey(sootMethod))
                clzMethod.put(sootMethod, new HashMap<>());
            else
                return clzMethod.get(sootMethod);

            HashMap<Integer, SootField> fieldWithParam = clzMethod.get(sootMethod);
            if (isStandardLibraryClass(clz)) return fieldWithParam;

            HashMap<Value, List<Integer>> localFromParam = new HashMap<>();
            HashMap<Value, MethodParamInvoke> localFromParamInvoke = new HashMap<>();
            HashMap<Value, SootField> localToFieldMap = new HashMap<>();
            HashMap<Value, List<String>> localToString = new HashMap<>();
            HashMap<Value, AbstractClz> localToCLz = new HashMap<>();
            HashMap<Value, List<List<String>>> localArray = new HashMap<>();

            if (visitedMethod.contains(sootMethod.getSignature())) return fieldWithParam;
            visitedMethod.add(sootMethod.getSignature());

            Body body = null;


            try {
                body = sootMethod.retrieveActiveBody();
            } catch (Exception e) {
                //LOGGER.error("Could not retrieved the active body {} because {}", sootMethod, e.getLocalizedMessage());
            }
            //LOGGER.info("THE METHOD NOW: {}", sootMethod);
            if (body != null) {
                for (Unit unit : body.getUnits()) {
                    if (unit.toString().contains("$r1 = staticinvoke <com.gooclient.anycam.activity.settings.update.MyHttp: java.lang.String encryption(java.lang.String,java.lang.String)>"))
                        LOGGER.info("ggg");
                    Stmt stmt = (Stmt) unit;
                    if (stmt instanceof IdentityStmt) {
                        Value rightOp = ((IdentityStmt) stmt).getRightOp();
                        Value leftOp = ((IdentityStmt) stmt).getLeftOp();
                        if (rightOp instanceof ParameterRef) {
                            int index = ((ParameterRef) rightOp).getIndex();
                            addValue(localFromParam, leftOp, index);
                            // LOGGER.info("275: localFromParam : {}-{}", leftOp, index);
                        }
                    } else if (stmt instanceof AssignStmt) {
                        Value rightOp = ((AssignStmt) stmt).getRightOp();
                        Value leftOp = ((AssignStmt) stmt).getLeftOp();
                        if (rightOp instanceof NewArrayExpr) {
                            NewArrayExpr arrayExpr = (NewArrayExpr) rightOp;
                            if (arrayExpr.getType().toString().contains("byte")) continue;
                            Integer index = (Integer) SimulateUtil.getConstant(arrayExpr.getSize());
                            if (index != null && index > 0 && index < 100) {
                                List<List<String>> arrayList = new ArrayList<>();
                                int i = 0;
                                while (i < index) {
                                    arrayList.add(new ArrayList<>());
                                    i++;
                                }
                                localArray.put(leftOp, arrayList);
                                //LOGGER.info("297: New local Array: {}, {}, {},{}", stmt, leftOp, arrayList, index);
                            } else {
                                localArray.remove(leftOp);
                            }
                        } else if (leftOp instanceof Local) {
                            // localArray Value assigned new value
                            //todo right is fieldREF, right is staticinvoke,.

                            if (localFromParam.containsKey(rightOp)) {
                                List<Integer> keys = localFromParam.get(rightOp);
                                localFromParam.put(leftOp, keys);
                            }

                            if (localFromParamInvoke.containsKey(rightOp)) {
                                localFromParamInvoke.put(leftOp, localFromParamInvoke.get(rightOp));
                                localArray.remove(leftOp);
                                localToString.remove(leftOp);
                                localFromParam.remove(leftOp); //update the value
                            }

                            if (rightOp instanceof Local) {
                                if (localToCLz.containsKey(rightOp))
                                    localToCLz.put(leftOp, localToCLz.get(rightOp));
                            } else if (rightOp instanceof CastExpr) {
                                Value op = ((CastExpr) rightOp).getOp();
                                if (localToCLz.containsKey(op))
                                    localToCLz.put(leftOp, localToCLz.get(op));
                                if (localFromParam.containsKey(op)) {
                                    if (!leftOp.equals(op))
                                        localFromParam.remove(leftOp);
                                    addValue(localFromParam, leftOp, localFromParam.get(op));
                                }
                            } else if (rightOp instanceof ArrayRef) {
                                ArrayRef arrayRef = (ArrayRef) rightOp;
                                Value arrayBase = arrayRef.getBase();
                                localArray.remove(leftOp);

                                Object obj = SimulateUtil.getConstant(arrayRef.getIndex());
                                if (obj instanceof Integer) {
                                    int arrayIndex = (Integer) obj;
                                    if (localArray.containsKey(arrayBase)) {
                                        try {
                                            List<List<String>> array = localArray.get(arrayBase);
                                            if (array.size() - 1 < arrayIndex) continue;
                                            List<String> arrayValue = array.get(arrayIndex);
                                            if (arrayValue.isEmpty()) continue;
                                            String invokeSigStr = null;
                                            List<Integer> arrayID = getArrayFieldIndex(arrayValue);
                                            if (!arrayID.isEmpty()) {
                                                if (arrayValue.get(0).startsWith("MethodParamInvoke")) {
                                                    invokeSigStr = getInvokeSigOfString(arrayValue.get(0));
                                                    List<String> tmp = new ArrayList<>(arrayValue);
                                                    tmp.set(0, Objects.requireNonNullElse(invokeSigStr, "null"));
                                                    MethodParamInvoke methodParamInvoke = new MethodParamInvoke(sootMethod, arrayID, tmp);
                                                    localFromParamInvoke.put(leftOp, methodParamInvoke);

                                                    localToString.remove(leftOp);
                                                    localFromParam.remove(leftOp);
                                                    localToCLz.remove(leftOp);
                                                } else {
                                                    List<Integer> params = new ArrayList<>();
                                                    for (int fromId : arrayID) {
                                                        if (localFromParam.containsKey(arrayBase)) {
                                                            if (localFromParam.get(arrayBase).contains(fromId)) {
                                                                addValue(params, fromId);
                                                            }
                                                        }
                                                    }
                                                    if (arrayValue.get(0).startsWith("<") && !params.isEmpty()) {
                                                        MethodParamInvoke methodParamInvoke = new MethodParamInvoke(sootMethod, params, arrayValue);
                                                        localFromParamInvoke.put(leftOp, methodParamInvoke);

                                                        localToString.remove(leftOp);
                                                        localFromParam.remove(leftOp);
                                                        localToCLz.remove(leftOp);
                                                    } else {
                                                        localFromParam.remove(leftOp);
                                                        addValue(localFromParam, leftOp, params);
                                                    }
                                                }
                                            } else {
                                                localFromParam.remove(leftOp); //update the value
                                                localToString.put(leftOp, arrayValue);
                                            }

                                        } catch (Exception e) {
                                            //LOGGER.error("325: array index over, {},{},{},{}", clz, sootMethod, unit, localArray.get(arrayBase));
                                        }
                                    } else if (localFromParamInvoke.containsKey(arrayBase)) {
                                        MethodParamInvoke methodParamInvoke = new MethodParamInvoke(localFromParamInvoke.get(arrayBase));
                                        methodParamInvoke.addMethodInvoke(arrayRef.toString());
                                        localFromParamInvoke.put(leftOp, methodParamInvoke);
                                        localFromParam.remove(leftOp); //update the value
                                    }
                                }
                            } else if (rightOp instanceof NewExpr) {
                                SootClass sootClz = ((NewExpr) rightOp).getBaseType().getSootClass();
                                String clzName = ((NewExpr) rightOp).getBaseType().getClassName();
                                if (MethodLocal.isCommonClz(clzName)) {
                                    AbstractClz abstractClz = MethodLocal.CreateCommonClz(sootClz, sootMethod);
                                    if (abstractClz != null)
                                        localToCLz.put(leftOp, abstractClz);
                                    localFromParamInvoke.remove(leftOp);
                                    localArray.remove(leftOp);
                                    localToString.remove(leftOp);
                                    localFromParam.remove(leftOp);

                                }
                                //TODO cla simulate

                            } else if (rightOp instanceof Constant) {
                                Object obj = SimulateUtil.getConstant(rightOp);
                                if (obj != null) {
                                    addValue(localToString, leftOp, obj.toString());
                                    localFromParamInvoke.remove(leftOp);

                                    SootField sootField = localToFieldMap.remove(leftOp);
                                    AbstractClz abstractClz = localToCLz.remove(leftOp);
                                    if (abstractClz != null && sootField != null) {
                                        abstractClz.solve();
                                        if (abstractClz.isSolved())
                                            addValue(fieldToString, sootField.toString(), abstractClz.toString());
                                    }
                                    localFromParam.remove(leftOp); //update the value
                                    localToCLz.remove(leftOp);

                                }
                            } else if (rightOp instanceof FieldRef) {
                                SootField field = ((FieldRef) rightOp).getField();
                                if (fieldToString.containsKey(field.toString())) {
                                    localToString.put(leftOp, fieldToString.get((field.toString())));
                                } else {
                                    addValue(localToString, leftOp, field.toString());
                                }
                                localFromParamInvoke.remove(leftOp);
                                localFromParam.remove(leftOp); //update the value
                                localArray.remove(leftOp);

                                SootField sootField = localToFieldMap.remove(leftOp);
                                AbstractClz abstractClz = localToCLz.remove(leftOp);
                                if (abstractClz != null && sootField != null) {
                                    abstractClz.solve();
                                    if (abstractClz.isSolved())
                                        addValue(fieldToString, sootField.toString(), abstractClz.toString());
                                }

                                if (field.getType() instanceof RefType) {
                                    RefType refType = (RefType) field.getType();
                                    if (refType.getClassName().contains("Map")) {
                                        localToFieldMap.put(leftOp, field);
                                        //Todo UPDATE FIELD STRING
                                        HashmapClz hashmapClz = new HashmapClz(refType.getSootClass(), sootMethod);
                                        localToCLz.put(leftOp, hashmapClz);

                                        localToString.remove(leftOp);
                                    }
                                }

                            } else if (rightOp instanceof InvokeExpr) {
                                try {
                                    SootMethod method = ((InvokeExpr) rightOp).getMethod();
                                    if ((methodToFieldString.containsKey(method) && !CallGraph.callBackClassesWithPoint.containsKey(sootMethod.getDeclaringClass())) || MethodString.getMethodToString().containsKey(method)) {
                                        if (methodToString.containsKey(method)) {
                                            localToString.put(leftOp, MethodString.getMethodToString().get(method));
                                        } else if (methodToFieldString.containsKey(method)) {
                                            if (fieldToString.containsKey(methodToFieldString.get(method))) {
                                                List<String> tmp = fieldToString.get(methodToFieldString.get(method));
                                                localToString.put(leftOp, tmp);
                                                addValue(methodToString, method, tmp);
                                            } else {
                                                localToString.put(leftOp, new ArrayList<>(List.of(methodToFieldString.get(method))));
                                            }
                                        }
                                        localFromParam.remove(leftOp);
                                        localFromParamInvoke.remove(leftOp);
                                        localToCLz.remove(leftOp);
                                    } else if (rightOp instanceof InstanceInvokeExpr) {
                                        Value rightBase = ((InstanceInvokeExpr) rightOp).getBase();
                                        Integer baseParam = -1;
                                        if (localFromParam.containsKey(rightBase))
                                            baseParam = localFromParam.get(rightBase).get(0);
                                        HashMap<Value, List<String>> paramValueWithStrings = new HashMap<>();
                                        HashMap<Integer, List<String>> paramIndexWithStrings = new HashMap<>();

                                        int index = 0;
                                        for (Value arg : ((InstanceInvokeExpr) rightOp).getArgs()) {
                                            if (localToCLz.containsKey(arg)) {
                                                AbstractClz abstractClz = localToCLz.get(arg);
                                                abstractClz.solve();
                                                if (abstractClz.isSolved()) {
                                                    paramValueWithStrings.put(arg, new ArrayList<>(List.of(abstractClz.toString())));

                                                }
                                                if (localFromParam.containsKey(arg)) {
                                                    addValue(localFromParam, rightBase, localFromParam.get(arg));
                                                }
                                            } else if (arg instanceof Constant) {
                                                Object obj = SimulateUtil.getConstant(arg);
                                                if (obj != null) {
                                                    String objString = obj.toString();
                                                    paramValueWithStrings.put(arg, new ArrayList<>(List.of(objString)));
                                                }
                                            } else if (localArray.containsKey(arg)) {
                                                if (localFromParam.containsKey(arg)) {
                                                    addValue(localFromParam, rightBase, localFromParam.get(arg));
                                                }
                                                paramValueWithStrings.put(arg, new ArrayList<>(List.of(getContent(localArray.get(arg)))));
                                            } else if (localToString.containsKey(arg)) {
                                                paramValueWithStrings.put(arg, localToString.get(arg));
                                            } else if (localFromParamInvoke.containsKey(arg)) {
                                                addValue(localFromParam, rightBase, localFromParamInvoke.get(arg).param);
                                                MethodParamInvoke methodParamInvoke = new MethodParamInvoke(localFromParamInvoke.get(arg));
                                                paramValueWithStrings.put(arg, new ArrayList<>(List.of(getContent(methodParamInvoke.invokeMethodSig))));
                                            } else if (localFromParam.containsKey(arg)) {
                                                addValue(localFromParam, rightBase, localFromParam.get(arg));
                                                paramValueWithStrings.put(arg, MethodString.paramListToString(localFromParam.get(arg)));
                                            }
                                            if (paramValueWithStrings.containsKey(arg))
                                                paramIndexWithStrings.put(index, paramValueWithStrings.get(arg));
                                            index++;

                                        }
                                        if (localToCLz.containsKey(rightBase)) {
                                            ValueContext valueContext = new ValueContext(sootMethod, unit, paramValueWithStrings);
                                            AbstractClz abstractClz = localToCLz.get(rightBase);
                                            abstractClz.addValueContexts(valueContext);
                                            abstractClz.solve();
                                            if (localToFieldMap.containsKey(rightBase) && abstractClz.isSolved())
                                                addValue(fieldToString, localToFieldMap.get(rightBase).toString(), new ArrayList<>(List.of(abstractClz.toString())));

                                            localToCLz.put(leftOp, localToCLz.get(rightBase));
                                            localFromParamInvoke.remove(leftOp);
                                            localArray.remove(leftOp);

                                            if (!leftOp.equals(rightBase)) {// assignment , leftOp = rightBase
                                                localFromParam.remove(leftOp);
                                                if (localFromParam.containsKey(rightBase)) {
                                                    addValue(localFromParam, leftOp, localFromParam.get(rightBase));
                                                }
                                            }

                                        } else if (localFromParamInvoke.containsKey(rightBase)) {
                                            MethodParamInvoke methodParamInvokeBase = localFromParamInvoke.get(rightBase);
                                            MethodParamInvoke methodParamInvoke = new MethodParamInvoke(sootMethod, methodParamInvokeBase.param, methodParamInvokeBase.invokeMethodSig);
                                            if (localFromParam.containsKey(rightBase))
                                                addValue(methodParamInvoke.param, localFromParam.get(rightBase));
                                            if (method.getDeclaringClass().getName().contains("String"))
                                                methodParamInvoke.addMethodInvoke(method.getName() + paramIndexWithStrings);
                                            else
                                                methodParamInvoke.addMethodInvoke(method.getSignature() + paramIndexWithStrings);
                                            localFromParamInvoke.put(leftOp, methodParamInvoke);
                                            localArray.remove(leftOp);
                                            localToString.remove(leftOp);
                                            localFromParam.remove(leftOp); //update the value
                                            localToCLz.remove(leftOp);
                                            localToFieldMap.remove(leftOp);

                                        } else if (localFromParam.containsKey(rightBase)) {
                                            String clzName = method.getDeclaringClass().getName();
                                            if (!clzName.contains("ViewBinding") && !clzName.contains("Dialog") && !clzName.contains("panel") && !clzName.contains("widget")) {
                                                MethodParamInvoke methodParamInvoke;
                                                if (baseParam > -1) {
                                                    methodParamInvoke = new MethodParamInvoke(sootMethod, localFromParam.get(rightBase), "Base=$[" + baseParam + "]");
                                                    if (method.getDeclaringClass().getName().contains("String"))
                                                        methodParamInvoke.addMethodInvoke(method.getName() + paramIndexWithStrings);
                                                    else
                                                        methodParamInvoke.addMethodInvoke(method.getSignature() + paramIndexWithStrings);
                                                } else {
                                                    methodParamInvoke = new MethodParamInvoke(sootMethod, localFromParam.get(rightBase), method.getSignature() + paramIndexWithStrings);
                                                }
                                                localFromParamInvoke.put(leftOp, methodParamInvoke);
                                                localArray.remove(leftOp);
                                                localFromParam.remove(leftOp);

                                                localToString.remove(leftOp);
                                                localToCLz.remove(leftOp);
                                                localToFieldMap.remove(leftOp);
                                            }
                                        }
                                    } else if (rightOp instanceof StaticInvokeExpr) {
                                        StaticInvokeExpr staticInvokeExpr = (StaticInvokeExpr) rightOp;
                                        HashMap<Integer, List<String>> paramIndexWithStrings = new HashMap<>();
                                        List<Integer> fromParamArg = new ArrayList<>();
                                        int index = 0;
                                        for (Value arg : staticInvokeExpr.getArgs()) {
                                            if (localToCLz.containsKey(arg)) {
                                                AbstractClz abstractClz = localToCLz.get(arg);
                                                abstractClz.solve();
                                                if (abstractClz.isSolved()) {
                                                    paramIndexWithStrings.put(index, new ArrayList<>(List.of(abstractClz.toString())));
                                                }
                                                if (localFromParam.containsKey(arg)) {
                                                    if (localToString.containsKey(arg)) {
                                                        paramIndexWithStrings.remove(index);
                                                        addValue(paramIndexWithStrings, index, localToString.get(arg));
                                                    } else
                                                        addValue(fromParamArg, localFromParam.get(arg));
                                                }
                                            } else if (arg instanceof Constant) {
                                                Object obj = SimulateUtil.getConstant(arg);
                                                if (obj != null) {
                                                    String objString = obj.toString();
                                                    paramIndexWithStrings.put(index, new ArrayList<>(List.of(objString)));
                                                }
                                            } else if (localArray.containsKey(arg)) {
                                                if (localFromParam.containsKey(arg)) {
                                                    addValue(fromParamArg, localFromParam.get(arg));
                                                }
                                                paramIndexWithStrings.put(index, new ArrayList<>(List.of(getContent(localArray.get(arg)))));
                                            } else if (localToString.containsKey(arg)) {
                                                paramIndexWithStrings.put(index, localToString.get(arg));
                                            } else if (localFromParamInvoke.containsKey(arg)) {
                                                addValue(fromParamArg, localFromParamInvoke.get(arg).param);
                                                MethodParamInvoke methodParamInvoke = new MethodParamInvoke(localFromParamInvoke.get(arg));
                                                paramIndexWithStrings.put(index, new ArrayList<>(methodParamInvoke.invokeMethodSig));
                                            } else if (localFromParam.containsKey(arg)) {
                                                addValue(fromParamArg, localFromParam.get(arg));
                                                paramIndexWithStrings.put(index, MethodString.paramListToString(localFromParam.get(arg)));
                                            }
                                            index++;
                                        }
                                        List<String> returnResult = MethodLocal.getStaticInvokeReturn(staticInvokeExpr, paramIndexWithStrings);
                                        MethodParamInvoke methodParamInvoke;
                                        if (!returnResult.isEmpty()) {
                                            if (fromParamArg.isEmpty()) {
                                                localToString.put(leftOp, returnResult);
                                                localFromParamInvoke.remove(leftOp);
                                                localArray.remove(leftOp);
                                                localFromParam.remove(leftOp);
                                                continue;
                                            } else {
                                                methodParamInvoke = new MethodParamInvoke(sootMethod, fromParamArg, returnResult);
                                            }
                                        } else {
                                            methodParamInvoke = new MethodParamInvoke(sootMethod, fromParamArg, method.getSignature() + getContent(paramIndexWithStrings));
                                            if (fromParamArg.isEmpty()) {
                                                methodParamInvoke.addParamValue(paramIndexWithStrings);
                                                methodParamInvoke.solve();
                                            }
                                            if (method.getReturnType() instanceof ArrayType) {
                                                ArrayType arrayType = (ArrayType) method.getReturnType();
                                                if (arrayType.toString().toLowerCase().contains("byte")) continue;
                                                int LEN = 10;
                                                List<List<String>> arrayList = new ArrayList<>();
                                                List<String> arrayValue = new ArrayList<>();
                                                if (fromParamArg.isEmpty() && !methodParamInvoke.invokeMethodSig.isEmpty())
                                                    arrayValue.add(getContent(methodParamInvoke.invokeMethodSig));
                                                else {
                                                    arrayValue.add(methodParamInvoke.toString());
                                                    localFromParam.put(leftOp, fromParamArg);
                                                }
                                                int i = 0;
                                                while (i < LEN) {
                                                    List<String> tmpValue = new ArrayList<>(arrayValue);
                                                    tmpValue.add("get(" + i + ")");
                                                    arrayList.add(tmpValue);
                                                    i++;
                                                }
                                                localArray.put(leftOp, arrayList);

                                                localFromParamInvoke.remove(leftOp);
                                                localToString.remove(leftOp);
                                                localToCLz.remove(leftOp);
                                                localToFieldMap.remove(leftOp);
                                                //LOGGER.info("297: New local Array: {}, {}, {},{}", stmt, leftOp, arrayList, index);

                                            }
                                        }
                                        if (!(method.getReturnType() instanceof ArrayType)) {
                                            localFromParamInvoke.put(leftOp, methodParamInvoke);
                                            localArray.remove(leftOp);
                                            localToString.remove(leftOp);
                                            localToCLz.remove(leftOp);
                                            localToFieldMap.remove(leftOp);
                                            localFromParam.remove(leftOp);
                                        }
                                        //update the value
                                    }

                                } catch (Exception e) {
                                    LOGGER.error(e);
                                }

                            }
                        } else if (leftOp instanceof FieldRef) {
                            FieldRef leftFiledRef = (FieldRef) leftOp;
                            SootField leftField = leftFiledRef.getField();
                            if (leftField.toString().contains("requestSign"))
                                LOGGER.info("got");
                            if (localToCLz.containsKey(rightOp) && !localFromParam.containsKey(rightOp)) {
                                AbstractClz abstractClz = localToCLz.get(rightOp);
                                abstractClz.solve();
                                if (abstractClz.isSolved())
                                    addValue(fieldToString, leftField.toString(), abstractClz.toString());
                            } else if (localToString.containsKey(rightOp)) {
                                addValue(fieldToString, leftField.toString(), localToString.get(rightOp));
                                //LOGGER.info("331:New Field with String: {}:{}; {}", leftField, localToString.get(rightOp), fieldToString.get(leftField.toString()).toString());
                                //localToString.remove(rightOp);
                                if (methodToFieldString.containsValue(leftField.toString())) {
                                    SootMethod method = getKeyByValue(methodToFieldString, leftField.toString());
                                    if (method != null) {
                                        methodToString.put(method, fieldToString.get(leftField.toString()));
                                        TryAddStringToInterface(method);
                                        //LOGGER.info("335:New Method To String: " + method.getSignature() + ":" + fieldToString.get(leftField.toString()) + ";" + methodToString.get(method).toString());
                                    }
                                }
                            } else if (rightOp instanceof Constant) {
                                if (rightOp instanceof NullConstant) continue;
                                Object obj = SimulateUtil.getConstant(rightOp);
                                if (obj == null) continue;
                                if (obj.toString().isEmpty()) continue;
                                addValue(fieldToString, leftField.toString(), obj.toString());
                                //LOGGER.info("New Field with String: " + leftField + ":" + rightOp + ";" + fieldToString.get(leftField.toString()).toString());
                                if (methodToFieldString.containsValue(leftField.toString())) {
                                    SootMethod method = getKeyByValue(methodToFieldString, leftField.toString());
                                    if (method != null) {
                                        addValue(methodToString, method, obj.toString());
                                        TryAddStringToInterface(method);
                                        //LOGGER.info("141:New Method To String: " + method.getSignature() + ":" + rightOp + ";" + methodToString.get(method).toString());
                                    }
                                }
                            }

                            if (localFromParamInvoke.containsKey(rightOp)) {
                                MethodParamInvoke methodParamInvoke = localFromParamInvoke.get(rightOp);
                                if (methodParamInvoke.param.isEmpty()) {
                                    addValue(fieldToString, leftField.toString(), getContent(methodParamInvoke.invokeMethodSig));
                                    LLMQuery.generate(methodParamInvoke);
                                } else
                                    addMethodInvokeSetField(leftField, methodParamInvoke);
                                if (leftField.getType() instanceof RefType) { //check class instance field
                                    String clzName = ((RefType) leftField.getType()).getClassName();
                                    if (CallGraph.outerInnerClasses.containsKey(clzName)) {
                                        List<SootField> allFields = classWithOuterInnerFields.get(clzName);
                                        for (SootField field : allFields) {
                                            addMethodInvokeSetField(field, new MethodParamInvoke(methodParamInvoke));
                                        }
                                    }
                                }
                            } else if (localArray.containsKey(rightOp)) {
                                if (!fieldToString.containsKey(leftField.toString()))
                                    fieldToString.put(leftField.toString(), new ArrayList<>());
                                addArrayStringTo(localArray.get(rightOp), fieldToString.get(leftField.toString()));
                                //LOGGER.info("337: New Field with Array String: {}==>{}--{}", localArray.get(rightOp), leftField, fieldToString.get(leftField.toString()));
                            } else if (localFromParam.containsKey(rightOp)) {
                                List<Integer> indexes = localFromParam.get(rightOp);
                                for (Integer index : indexes) {
                                    if (leftField.toString().contains("panel") || leftField.toString().contains("ViewBinding"))
                                        continue;
                                    fieldWithParam.put(index, leftField);
                                }

                                MethodParamInvoke methodParamInvoke = new MethodParamInvoke(sootMethod, indexes);
                                if (localToCLz.containsKey(rightOp)) {
                                    methodParamInvoke.addMethodInvoke(localToCLz.get(rightOp).toString());
                                    addMethodInvokeSetField(leftField, methodParamInvoke);
                                }

                                if (!indexes.isEmpty() && leftField.getType() instanceof RefType) {
                                    String clzName = ((RefType) leftField.getType()).getClassName();
                                    if (CallGraph.outerInnerClasses.containsKey(clzName)) {
                                        methodParamInvoke.addMethodInvoke(getContent(paramListToString(indexes)));
                                        List<SootField> allFields = classWithOuterInnerFields.get(clzName);
                                        for (SootField field : allFields) {
                                            if (isStandardLibraryClass(field.getDeclaringClass())) continue;
                                            if (field.toString().contains("panel") || field.toString().contains("ViewBinding") || field.toString().contains("this$0"))
                                                continue;
                                            addMethodInvokeSetField(field, methodParamInvoke);
                                        }
                                    }
                                }

                            }


                        } else if (leftOp instanceof ArrayRef) {
                            ArrayRef arrayRef = (ArrayRef) leftOp;
                            Integer arrayIndex = (Integer) SimulateUtil.getConstant(arrayRef.getIndex());
                            Value base = arrayRef.getBase();
                            if (localArray.containsKey(base)) {
                                if (arrayIndex != null) {
                                    try {
                                        List<String> arrayValue = localArray.get(base).get(arrayIndex);

                                        if (rightOp instanceof Constant) {
                                            Object obj = SimulateUtil.getConstant(rightOp);
                                            if (obj != null)
                                                addValue(arrayValue, obj.toString());
                                        } else if (localToString.containsKey(rightOp)) {
                                            addValue(arrayValue, localToString.get(rightOp));
                                        } else if (localToCLz.containsKey(rightOp)) {
                                            AbstractClz abstractClz = localToCLz.get(rightOp);
                                            abstractClz.solve();
                                            if (abstractClz.isSolved()) {
                                                arrayValue.clear();
                                                arrayValue.add(abstractClz.toString());
                                            }
                                            if (localFromParam.containsKey(rightOp)) {
                                                addValue(localFromParam, base, localFromParam.get(rightOp));
                                            }
                                        } else if (localFromParam.containsKey(rightOp)) {
                                            arrayValue.clear();
                                            arrayValue.addAll(MethodString.paramListToString(localFromParam.get(rightOp)));
                                            addValue(localFromParam, base, localFromParam.get(rightOp));
                                        } else if (localFromParamInvoke.containsKey(rightOp)) {
                                            MethodParamInvoke methodParamInvoke = new MethodParamInvoke(localFromParamInvoke.get(rightOp));
                                            addValue(localFromParam, base, methodParamInvoke.param);
                                            arrayValue.clear();
                                            arrayValue.addAll(new ArrayList<>(methodParamInvoke.invokeMethodSig));

                                        }
                                    } catch (Exception e) {
                                        //LOGGER.error("348: array index over, {},{},{},{}", clz, sootMethod, unit, localArray.get(base));
                                    }
                                }
                            } else {
                                //TODO
                            }
                        }
                    } else if (stmt instanceof InvokeStmt) {
                        //LOGGER.info("291: Invoke: {}", stmt);
                        boolean isNeedInvoke = false;
                        Value base = null;
                        if (stmt.getInvokeExpr() instanceof InstanceInvokeExpr)
                            base = ((InstanceInvokeExpr) stmt.getInvokeExpr()).getBase();
                        HashMap<Integer, Constant> paramWithConstant = new HashMap<>();
                        ValueContext valueContext = new ValueContext(sootMethod, unit);
                        //TODO check method solved.
                        HashMap<Integer, Value> localToParam = new HashMap<>();
                        int index = 0;
                        for (Value param : stmt.getInvokeExpr().getArgs()) {
                            if (param instanceof Constant) {
                                paramWithConstant.put(index, (Constant) param);
                                //LOGGER.info("304: param constant: {}--{}-{}",stmt.getInvokeExpr(),param, index);
                                isNeedInvoke = true;
                            } else if (localToString.containsKey(param)) {
                                isNeedInvoke = true;
                                localToParam.put(index, param);
                                paramWithConstant.put(index, null);
                                addValue(valueContext.getCurrentValues(), param, localToString.get(param));
                            } else if (localArray.containsKey(param)) {
                                isNeedInvoke = true;
                                List<String> paramList = new ArrayList<>();
                                for (List<String> strs : localArray.get(param)) {
                                    paramList.add(getContent(strs));
                                }
                                localToParam.put(index, param);
                                localToString.put(param, paramList);
                                paramWithConstant.put(index, null);
                                addValue(valueContext.getCurrentValues(), param, localToString.get(param));
                            } else if (localToCLz.containsKey(param)) {
                                isNeedInvoke = true;
                                addValue(valueContext.getCurrentValues(), param, localToCLz.get(param).toString());
                                if (localFromParam.containsKey(param)) {
                                    if (base != null && localToCLz.containsKey(base)) {
                                        addValue(localFromParam, base, localFromParam.get(param));
                                    }
                                }
                            } else if (localFromParam.containsKey(param)) {
                                isNeedInvoke = true;
                                localToParam.put(index, param);
                                addValue(valueContext.getCurrentValues(), param, paramListToString(localFromParam.get(param)));

                                if (base != null && localToCLz.containsKey(base)) {
                                    addValue(localFromParam, base, localFromParam.get(param));
                                }
                                //LOGGER.info("301: localFromParam: {},{}", param, index);
                            } else if (localFromParamInvoke.containsKey(param)) {
                                isNeedInvoke = true;
                                localToParam.put(index, param);

                                if (base != null && localToCLz.containsKey(base)) {
                                    addValue(valueContext.getCurrentValues(), param, getContent(localFromParamInvoke.get(param).invokeMethodSig));
                                    addValue(localFromParam, base, localFromParamInvoke.get(param).param);
                                } else {
                                    addValue(valueContext.getCurrentValues(), param, localFromParamInvoke.get(param).toString()); //todo
                                }
                            }

                            index++;
                        }
                        if (isNeedInvoke) {
                            SootMethod invokeMethod = stmt.getInvokeExpr().getMethod();
                            HashMap<Integer, SootField> invokeResult = null;
                            if (invokeMethod.getDeclaringClass().equals(clz)) {
                                //LOGGER.info("311: Go to Invoke: {}", invokeMethod);
                                invokeResult = GetSetField(invokeMethod);
                                if (invokeResult.isEmpty()) {
                                    clzMethod.remove(invokeMethod);
                                }
                                //LOGGER.info("313: Get Invoke result: {}", invokeResult);
                            } else if (base != null && localToCLz.containsKey(base)) {
                                invokeResult = null;
                                AbstractClz abstractClz = localToCLz.get(base);
                                abstractClz.addValueContexts(valueContext);
                                abstractClz.solve();
                                if (localToFieldMap.containsKey(base) && abstractClz.isSolved())
                                    addValue(fieldToString, localToFieldMap.get(base).toString(), new ArrayList<>(List.of(abstractClz.toString())));
                            } else if(invokeMethod.getSignature().contains("java.lang.String replace")){
                                if(localToString.containsKey(base) && paramWithConstant.size()==2){
                                    List<String> tmpResult = localToString.get(base);
                                    Object obj0 = SimulateUtil.getConstant(paramWithConstant.get(0));
                                    Object obj1 = SimulateUtil.getConstant(paramWithConstant.get(1));
                                    if(obj0 != null && obj1 != null)
                                        tmpResult.replaceAll(s -> s.replace(obj0.toString(), obj1.toString()));
                                    invokeResult = null;
                                }
                            }else{
                                HashMap<SootMethod, HashMap<Integer, SootField>> invokeClzResult;
                                //LOGGER.info("316: Go to Clz :{}", invokeMethod.getDeclaringClass().getName());
                                invokeClzResult = GetMethodSetField(invokeMethod.getDeclaringClass());
                                invokeResult = invokeClzResult.get(invokeMethod);
                                //LOGGER.info("319: Get Clz result:{}", invokeResult);
                            }
                            HashMap<SootField, List<MethodParamInvoke>> tempMethodInvoke = new HashMap<>();
                            //Check if invoke method has been analyzed, and merge the method with invoke method by param transfer.
                            if (methodToInvokeSetFields.containsKey(invokeMethod)) {
                                for (SootField field : methodToInvokeSetFields.get(invokeMethod)) {
                                    List<MethodParamInvoke> methodParamInvokes = FieldSetByMethodInvoke.get(field);
                                    for (MethodParamInvoke methodParamInvoke : methodParamInvokes) {
                                        if (!methodParamInvoke.sootMethod.equals(invokeMethod)) continue;
                                        for (Integer paramIndex : methodParamInvoke.param) {
                                            if (paramWithConstant.containsKey(paramIndex)) {
                                                if (paramWithConstant.get(paramIndex) != null) {
                                                    Object obj = SimulateUtil.getConstant(paramWithConstant.get(paramIndex));
                                                    if (obj != null)
                                                        methodParamInvoke.addParamValue(paramIndex, obj.toString());
                                                }
                                            }
                                            if (localToParam.containsKey(paramIndex)) {
                                                Value localValue = localToParam.get(paramIndex);
                                                if (localFromParamInvoke.containsKey(localValue)) {
                                                    MethodParamInvoke methodInvoke = localFromParamInvoke.get(localValue);
                                                    methodInvoke.addMethodInvoke(methodParamInvoke.invokeMethodSig);
                                                    //todo more analysis. maybe del the original methodParamInvoke.
                                                    addValue(tempMethodInvoke, field, methodInvoke);
                                                    //LOGGER.warn("509: GetMethodInvokeSetField: {} -> {}", field, methodInvokeSetField.get(field).toString());
                                                } else if (localToString.containsKey(localValue)) {
                                                    methodParamInvoke.addParamValue(paramIndex, localToString.get(localValue));
                                                } else if (localArray.containsKey(localValue)) {
                                                    methodParamInvoke.addParamValue(paramIndex, getContent(localArray.get(localValue)));
                                                } else if (localToCLz.containsKey(localValue)) {
                                                    methodParamInvoke.addParamValue(paramIndex, localToCLz.get(localValue).toString());
                                                } else {
                                                    List<Integer> indexes = localFromParam.get(localValue);
                                                    if (indexes.isEmpty()) continue;
                                                    String clzName = sootMethod.getDeclaringClass().getName();
                                                    if (!clzName.contains("ViewBinding") && !clzName.contains("Dialog") && !clzName.contains("panel") && !clzName.contains("widget")) {
                                                        MethodParamInvoke newMethodParamInvoke = new MethodParamInvoke(sootMethod, indexes, methodParamInvoke.invokeMethodSig);
                                                        addValue(tempMethodInvoke, field, newMethodParamInvoke);
                                                        //LOGGER.warn("515: GetMethodInvokeSetField: {} -> {}", field, methodInvokeSetField.get(field).toString());
                                                    }
                                                }
                                            }
                                        }
                                    }
                                }
                            }

                            for (SootField field : tempMethodInvoke.keySet()) {
                                for (MethodParamInvoke methodParamInvoke : tempMethodInvoke.get(field))
                                    addMethodInvokeSetField(field, methodParamInvoke);
                            }

                            //invoke Result only contains that param to field directly
                            if (invokeResult != null) {
                                if (!invokeResult.isEmpty() && !localToParam.isEmpty()) {
                                    for (Integer id : localToParam.keySet()) {
                                        if (invokeResult.containsKey(id)) {
                                            SootField field = invokeResult.get(id);
                                            if (localFromParamInvoke.containsKey(localToParam.get(id))) {
                                                MethodParamInvoke methodParamInvoke = localFromParamInvoke.get(localToParam.get(id));
                                                addMethodInvokeSetField(field, methodParamInvoke);
                                                // LOGGER.warn("532: GetMethodInvokeSetField: {} -> {}", field, methodInvokeSetField.get(field).toString());
                                            } else if (localFromParam.containsKey(localToParam.get(id))) {
                                                List<Integer> indexes = localFromParam.get(localToParam.get(id));
                                                for (Integer paramIndex : indexes) {
                                                    if (field.toString().contains("panel") || field.toString().contains("ViewBinding"))
                                                        continue;
                                                    fieldWithParam.put(paramIndex, field);
                                                }
                                                //LOGGER.info("309: GetSetField: {}--{}--{}",clz, invokeMethod, fieldWithParam);
                                            }
                                        }

                                    }

                                }
                                if (!paramWithConstant.isEmpty()) {
                                    //LOGGER.info("331: Constant param: Method: {}-- Result: {}-- constant: {}", invokeMethod,invokeResult, paramWithConstant);
                                    for (Integer constIndex : paramWithConstant.keySet()) {
                                        if (invokeResult.containsKey(constIndex)) {
                                            SootField field = invokeResult.get(constIndex);
                                            if (paramWithConstant.get(constIndex) == null) {
                                                for (String str : localToString.get(localToParam.get(constIndex))) {
                                                    if (str.equals(field.toString())) continue;
                                                    addValue(fieldToString, field.toString(), str);
                                                    //LOGGER.info("434: New local param To field: {} <= {}", field.toString(), str);
                                                }
                                            } else {
                                                Object constObj = SimulateUtil.getConstant(paramWithConstant.get(constIndex));
                                                if (constObj.toString().isEmpty()) continue;
                                                if (field != null && constObj != null) {
                                                    if (!fieldToString.containsKey(field.toString()))
                                                        fieldToString.put(field.toString(), new ArrayList<>());
                                                    List<String> fieldValues = fieldToString.get(field.toString());
                                                    List<Integer> arrayID = getArrayFieldIndex(fieldValues);
                                                    if (!arrayID.isEmpty()) {
                                                        for (int fromId : arrayID) {
                                                            if (constIndex.equals(fromId)) {
                                                                fieldValues.replaceAll(s -> s.replace("$[" + fromId + "]", constObj.toString()));
                                                                //LOGGER.info("445: New Array Param To field: {}, {}", fieldValues, arrayID);
                                                            }
                                                        }
                                                    } else {
                                                        addValue(fieldToString, field.toString(), constObj.toString());
                                                        if (methodToFieldString.containsValue(field.toString())) {
                                                            SootMethod method = getKeyByValue(methodToFieldString, field.toString());
                                                            if (method != null) {
                                                                methodToString.put(method, fieldToString.get(field.toString()));
                                                                TryAddStringToInterface(method);
                                                                //LOGGER.info("488:New Method To String: " + method.getSignature() + ":" + fieldToString.get(field.toString()) + ";" + methodToString.get(method).toString());
                                                            }
                                                        }
                                                    }
                                                    //LOGGER.info("320: New Field with String: " + field + ":" + constObj + ";" + fieldToString.get(field.toString()).toString());
                                                }
                                            }
                                        }
                                    }
                                }
                            } else if (index == 2 && !paramWithConstant.isEmpty()) {

                                boolean maybeCache = false;
                                String[] str = new String[2];
                                int num = 0;
                                Constant ob0 = paramWithConstant.get(0);

                                if (ob0 != null) {
                                    maybeCache = true;
                                    Object ob = SimulateUtil.getConstant(ob0);
                                    if (ob != null)
                                        str[0] = ob.toString();
                                    else
                                        maybeCache = false;
                                } else continue;

                                if (paramWithConstant.containsKey(1)) {
                                    Constant ob1 = paramWithConstant.get(1);
                                    if (ob1 == null)
                                        str[1] = getContent(localToString.get(localToParam.get(1)));
                                    else {
                                        Object ob = SimulateUtil.getConstant(ob1);
                                        if (ob != null)
                                            str[1] = ob.toString();
                                        else
                                            maybeCache = false;
                                    }
                                } else if (localToParam.containsKey(1)) {
                                    MethodParamInvoke methodParamInvoke = localFromParamInvoke.get(localToParam.get(1));
                                    if (methodParamInvoke != null && methodToFieldString.containsKey(methodParamInvoke.sootMethod)) {
                                        maybeCache = true;
                                        str[1] = methodParamInvoke.toString();
                                    } else maybeCache = false;
                                }
                                if (maybeCache && str[0] != null && str[1] != null) {
                                    SootClass invokeClz = invokeMethod.getDeclaringClass();
                                    if (!isStandardLibraryClass(invokeClz)) {
                                        if (!classMaybeCache.containsKey(invokeClz))
                                            classMaybeCache.put(invokeClz, new HashMap<>());
                                        addValue(classMaybeCache.get(invokeClz), str[0], str[1]);
                                        //LOGGER.warn("[ClassMaybeCache]: {}", classMaybeCache.get(invokeClz));
                                    }
                                }
                            }
                        }
                    } else if (stmt instanceof ReturnStmt) {
                        Value returnOP = ((ReturnStmt) stmt).getOp();
                        String returnType = sootMethod.getReturnType().toString().toLowerCase();
                        if (localFromParamInvoke.containsKey(returnOP)) {
                            if (methodReturnParamInvoke.containsKey(sootMethod)) {
                                if (methodReturnParamInvoke.get(sootMethod).invokeMethodSig.toString().length() < localFromParamInvoke.get(returnOP).invokeMethodSig.toString().length())
                                    methodReturnParamInvoke.put(sootMethod, localFromParamInvoke.get(returnOP));
                            } else {
                                methodReturnParamInvoke.put(sootMethod, localFromParamInvoke.get(returnOP));
                            }
                        } else if (localFromParam.containsKey(returnOP) && localToCLz.containsKey(returnOP)) {
                            AbstractClz abstractClz = localToCLz.get(returnOP);
                            abstractClz.solve();
                            if (abstractClz.isSolved()) {
                                MethodParamInvoke methodParamInvoke = new MethodParamInvoke(sootMethod, localFromParam.get(returnOP), abstractClz.toString());
                                if (methodReturnParamInvoke.containsKey(sootMethod)) {
                                    if (methodReturnParamInvoke.get(sootMethod).invokeMethodSig.toString().length() < methodParamInvoke.invokeMethodSig.toString().length())
                                        methodReturnParamInvoke.put(sootMethod, methodParamInvoke);
                                } else {
                                    methodReturnParamInvoke.put(sootMethod, methodParamInvoke);
                                }
                            }
                        }
                        if (localToString.containsKey(returnOP)) {
                            addValue(methodToString, sootMethod, localToString.get(returnOP));
                            if (returnType.contains("url") || returnType.contains("string")) {
                                String returnStr = localToString.get(returnOP).toString();
                                if (returnStr.startsWith("<") && returnStr.endsWith(">"))
                                    continue;
                                if (isFirmRelatedUrl(returnStr))
                                    LOGGER.debug("FIND local URL: {} ,from method {} ", returnStr, sootMethod);
                            }
                        }
                    }
                }
            }
            return fieldWithParam;
        }
        catch (Exception e){
            LOGGER.error("Exception in GetAllMethodSetField: {}", e.getMessage());
            return new HashMap<>();
        }
    }

    public static List<String> paramListToString(List<Integer> params){
        List<String> result = new ArrayList<>();
        for(Integer i : params){
            result.add("$[" + i + "]");
        }
        return result;
    }


    public static boolean isFirmRelatedUrl(String strParam){

        int i = 0;
        String str = strParam.toLowerCase();
        if((str.contains(".com") || str.contains(".net") || str.contains(".cn")) && (str.contains("://") || str.contains("http")))
            i = i + 2;
        if(str.contains("firmware") || str.contains("upgrade") || str.contains("update") || (str.contains("version") && (str.contains("match") || str.contains("new")))){
            i++;
        }
        if((str.contains(".bin") || str.contains(".fw") || str.contains(".ota") || str.contains(".img")) && str.contains("/"))
            i = i + 2;
        if(str.contains("appupdate") || str.contains("app_update") || str.contains("!"))
            i--;
        if(strParam.startsWith("<") && strParam.endsWith(">"))
            i = i - 2;
        return i>2;
    }

    public static boolean isCommonType(Type type) {
        if (type.equals(IntType.v()) ||
                type.equals(BooleanType.v()) ||
                type.equals(FloatType.v()) ||
                type.equals(DoubleType.v()) ||
                type.equals(LongType.v()) ||
                type.equals(ShortType.v()) ||
                type.equals(ByteType.v())) {
            return true;
        }

        if (type instanceof RefType) {
            RefType refType = (RefType) type;
            String className = refType.getClassName();
            return className.equals("java.lang.String") ||
                    className.contains("Map") ||
                    className.equals("java.util.List") ||
                    className.equals("java.util.Set") ||
                    className.contains("Collection") ||
                    className.toLowerCase().contains("url");
        }

        return false;
    }

    public static boolean isStandardLibraryClass(SootClass sc) {
        String className = sc.getName().toLowerCase();
        for(String str: library) {
            if (className.contains(str))
                return true;
        }
        return className.startsWith("java.") || className.startsWith("javax.")
                || className.startsWith("android.") || className.startsWith("androidx.")||className.startsWith("kotlin.") || className.startsWith("rx.")
                 || className.startsWith("okhttp3") || className.startsWith("retrofit2.") || className.startsWith("org.");
    }

    public static boolean isStandardLibraryField(String Fieldstr) {
        String fieldString = Fieldstr.toLowerCase();
        for(String str: library) {
            if (fieldString.contains(str))
                return true;
        }
        return fieldString.startsWith("<java.") || fieldString.startsWith("<javax.")
                || fieldString.startsWith("<android.") || fieldString.startsWith("<androidx.")||fieldString.startsWith("<kotlin.") || fieldString.startsWith("<rx.")
                || fieldString.startsWith("<okhttp3") || fieldString.startsWith("<retrofit2.");
    }

    public static List<String> library = List.of("bumptech", "com.xiaomi", "com.android","com.google", "com.amazonaws", "thingclips", "okio", "alipay.sdk", "com.huawei","com.tuya.", "com.tencent",
            "org.apache","jsonobjectparent","io.", "facebook", "crashsdk", "cn.jpush", "cn.jiguang", "eventbus", "bouncycastle", "spongycastle", "openssl", "com.stripe", "com.blankj", "com.alibaba",
            "com.amap.","com.scwang","mikephil", "dialog","umeng", "panel", "airbnb", "freshchat", "webview", "widget", "com.baidu", ".video.", "com.meizu", "cn.sharesdk","com.taobao.", "greenrobot", ".push.", ".channel.", ".android.",".crash.");

    public static <K, V> K getKeyByValue(Map<K, V> map, V value) {
        for (Map.Entry<K, V> entry : map.entrySet()) {
            if (value.equals(entry.getValue())) {
                return entry.getKey();
            }
        }
        return null;
    }

    public static void addMethodInvokeSetField(SootField field, MethodParamInvoke methodParamInvoke){
        boolean needAdd = true;
        if(isStandardLibraryClass(field.getDeclaringClass())) return;
        if(field.toString().contains("panel") || field.toString().contains("ViewBinding") || field.toString().contains("this$0")) return;
        if(!FieldSetByMethodInvoke.containsKey(field))
            FieldSetByMethodInvoke.put(field, new ArrayList<>());
        for(MethodParamInvoke m : FieldSetByMethodInvoke.get(field))
            if(m.equals(methodParamInvoke)) {
                needAdd = false;
                break;
            }
        if(needAdd) {
            FieldSetByMethodInvoke.get(field).add(methodParamInvoke);
            SootMethod method = methodParamInvoke.sootMethod;
            if(!methodToInvokeSetFields.containsKey(method)){
                methodToInvokeSetFields.put(method, new HashSet<>());
            }
            methodToInvokeSetFields.get(method).add(field);
        }
    }

    public static <K, V> List<K> getAllKeyByValue(Map<K, V> map, V value) {
        List<K> result = new ArrayList<>();
        for (Map.Entry<K, V> entry : map.entrySet()) {
            if (value.equals(entry.getValue())) {
                result.add(entry.getKey());
            }
        }
        return result;
    }

    public static List<Integer> getArrayFieldIndex(List<String> values){
        String regex = "\\$\\[(\\d+)]|param=\\[(\\d+)(?:,\\s*(\\d+))*]";
        List<Integer> result = new ArrayList<>();
        Pattern pattern = Pattern.compile(regex);
        Matcher matcher = pattern.matcher(values.toString());

        while (matcher.find()) {
            // 将匹配的结果添加到列表中
            for (int i = 1 ; i <= matcher.groupCount(); i++) {
                if(matcher.group(i) != null) {
                    String str = matcher.group(i);
                    addValue(result, Integer.parseInt(str));
                }
            }
        }
        return result;
    }

    public static String getInvokeSigOfString(String str){
        String regex = "invokeMethodSig=(\\[(<.*?)]|(<.*?))}$";
        Pattern pattern = Pattern.compile(regex);
        Matcher matcher = pattern.matcher(str);

        while (matcher.find()) {
            // 将匹配的结果添加到列表中
            if(matcher.group(1) != null)
                return matcher.group(1);
            else if(matcher.group(2) != null)
                return matcher.group(2);
        }
        return null;
    }

    public static void addArrayStringTo(List<List<String>> arrayString, List<String> value){
        int max = 0;
        for(List<String> list: arrayString){
            if(list.size() > max)
                max = list.size();
        }
        for(int i=0; i< max; i++){
            for(List<String> list: arrayString){
                if(i<list.size()){
                    value.add(list.get(i));
                }
                else{
                    value.add("");
                }
            }
        }
    }

    public static HashMap<SootMethod,List<String>> getMethodToString(){
        return methodToString;
    }

    public static HashMap<SootMethod,String> getMethodToFieldString(){
        return methodToFieldString;
    }

    public static HashMap<String,List<String>> getFieldToString(){
        return fieldToString;
    }

    public static String getContent(Object obj) {
        try {
            if (obj instanceof List<?>) {
                List<?> list = (List<?>) obj;
                if (list.isEmpty()) return "";
                if (list.size() == 1) {
                    Object firstElement = list.get(0);
                    return getContent(firstElement);
                } else {
                    // 返回 List 的 toString() 形式
                    List<String> stringList = new ArrayList<>();
                    for (Object element : list) {
                        stringList.add(getContent(element));
                    }
                    return stringList.toString();
                }
            } else if (obj instanceof Map<?, ?>) {
                Map<?, ?> map = (Map<?, ?>) obj;
                StringBuilder sb = new StringBuilder("{");
                boolean first = true;
                for (Map.Entry<?, ?> entry : map.entrySet()) {
                    if (!first) sb.append(", ");
                    sb.append(getContent(entry.getKey())).append("=").append(getContent(entry.getValue()));
                    first = false;
                }
                sb.append("}");
                return sb.toString();
            } else if (obj instanceof String) {
                return (String) obj;
            } else if (obj == null) {
                return "null";
            } else {
                return obj.toString();
            }
        }
        catch (Exception e){
            return "";
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

    public static <K, V> void addValue(Map<K, List<V>> map, K key, V value) {
        map.computeIfAbsent(key, k -> new ArrayList<>());
        List<V> values = map.get(key);
        if(!values.contains(value)){
            values.add(value);
        }
    }

    public static <K, V> void addValue(Map<K, List<V>> map, K key, List<V> values) {
        map.computeIfAbsent(key, k -> new ArrayList<>());
        List<V> key_values = map.get(key);
        for(V value: values) {
            if(value!=null) {
                if (!key_values.contains(value)) {
                    key_values.add(value);
                }
            }
        }
    }

    public static <K, V> void addValue(Map<K, List<V>> map, Map<K , List<V>> map1) {
        for(K key1 : map1.keySet()) {
            if(map1.get(key1) == null || map1.get(key1).isEmpty()) continue;
            map.computeIfAbsent(key1, k -> new ArrayList<>());
            List<V> key_values = map1.get(key1);
            addValue(map.get(key1), key_values);
        }
    }

    public static void removeDuplicates(List<String> list) {
        HashSet<String> seen = new HashSet<>();
        list.removeIf(item -> !seen.add(item));
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

}
