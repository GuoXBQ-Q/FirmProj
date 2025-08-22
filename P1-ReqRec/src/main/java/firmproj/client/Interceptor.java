package firmproj.client;

import firmproj.base.MethodLocal;
import firmproj.base.MethodString;
import firmproj.objectSim.SimulateUtil;
import org.apache.logging.log4j.LogManager;
import org.apache.logging.log4j.Logger;
import soot.*;
import soot.jimple.*;

import java.awt.image.LookupOp;
import java.lang.reflect.Method;
import java.util.*;
import java.util.stream.Collectors;

public class Interceptor {
    public SootClass sootClass;
    public SootMethod currentMethod;
    public boolean isNotCustomInterceptor = false;
    public  HashMap<List<String>, List<String>> headers = new HashMap<>();

    private static final Logger LOGGER = LogManager.getLogger(Interceptor.class);

    public Interceptor(SootClass clz){
        this.sootClass = clz;
    }

    public String getResult(){
        StringBuilder result = new StringBuilder("Headers=");
        if(!headers.isEmpty()) {
            result.append(MethodString.getContent(headers));
            return result.toString();
        }
        else return "";
    }

    @Override
    public String toString() {
        return "Interceptor{" +
                "sootClass=" + sootClass +
                ", isNotCustomInterceptor=" + isNotCustomInterceptor +
                ", headers=" + MethodString.getContent(headers) +
                '}';
    }

    public void init(){
        if(sootClass.getName().startsWith("okhttp3.internal")){
            isNotCustomInterceptor = true;
        }
        else{
            for (SootMethod sootMethod : sootClass.getMethods()) {
                if (!sootMethod.isConcrete())
                    continue;
                if (!sootMethod.getSignature().contains("okhttp3.Response intercept(okhttp3.Interceptor$Chain)"))
                    continue;
                Body body = null;
                try {
                    body = sootMethod.retrieveActiveBody();
                } catch (Exception e) {
                    LOGGER.error("Could not retrieved the active body {} because {}", sootMethod, e.getLocalizedMessage());
                }
                if (body == null)
                    continue;
                HashMap<Value, HashMap<List<String>, List<String>>> localToMap = new HashMap<>();
                HashMap<Value, List<String>> localToString = new HashMap<>();
                for (Unit unit : body.getUnits()) {
                    if (unit instanceof AssignStmt){
                        Value rightOp = ((AssignStmt) unit).getRightOp();
                        Value leftOp = ((AssignStmt) unit).getLeftOp();
                        if (rightOp instanceof NewExpr){
                            NewExpr newExpr = (NewExpr) rightOp;
                            String clsName = newExpr.getBaseType().getClassName();
                            if(clsName.contains("JSONObject") || clsName.contains("HashMap")){
                                localToMap.put(leftOp, new HashMap<>());
                                LOGGER.info("New local Map: {}", localToMap);
                            }
                        }
                        else if(leftOp instanceof Local){
                            if(rightOp instanceof FieldRef){
                                if(MethodString.getFieldToString().containsKey(((FieldRef) rightOp).getField().toString())){
                                    localToString.put(leftOp, MethodString.getFieldToString().get(((FieldRef) rightOp).getField().toString()));
                                }
                                else{
                                    localToString.put(leftOp, List.of(((FieldRef) rightOp).getField().toString()));
                                }
                            }else if(rightOp instanceof Constant){
                                Object obj = SimulateUtil.getConstant(rightOp);
                                if(obj != null)
                                    localToString.put(leftOp, List.of(obj.toString()));
                            }
                            else if(rightOp instanceof InvokeExpr){
                                InvokeExpr invokeExpr = (InvokeExpr)rightOp;
                                SootMethod method = ((InvokeExpr) rightOp).getMethod();
                                if(method.getSignature().contains("toString") && rightOp instanceof InstanceInvokeExpr){
                                    Value base = ((InstanceInvokeExpr) rightOp).getBase();
                                    if(localToMap.containsKey(base)){
                                        List<String> list = new ArrayList<>(List.of(MethodString.getContent(localToMap.get(base))));
                                        localToString.put(leftOp, list);
                                    }
                                    else if(localToString.containsKey(base))
                                        localToString.put(leftOp, localToString.get(base));
                                }
                                else if(method.getSignature().contains("internal.Boxing") || method.getName().contains("valueOf")){
                                    if(localToString.containsKey(invokeExpr.getArg(0)))
                                        localToString.put(leftOp, localToString.get(invokeExpr.getArg(0)));
                                }else if(MethodString.getMethodToString().containsKey(method)){
                                    localToString.put(leftOp, MethodString.getMethodToString().get(method));
                                }
                                else if(MethodString.getMethodToFieldString().containsKey(method)){
                                    localToString.put(leftOp, List.of(MethodString.getMethodToFieldString().get(method)));
                                }

                                else if(method.getSignature().contains("okhttp3.Request$Builder: okhttp3.Request$Builder addHeader")){
                                    List<List<String>> argString = new ArrayList<>();
                                    for(int i=0;i<2;i++){
                                        Value arg = invokeExpr.getArg(i);
                                        if(arg instanceof Constant){
                                            Object obj = SimulateUtil.getConstant(arg);
                                            if(obj != null)
                                                argString.add(List.of(obj.toString()));
                                        }
                                        else if(localToString.containsKey(arg)){
                                            argString.add(localToString.get(arg));
                                        }
                                        else
                                            argString.add(List.of("UNKNOWN"));
                                    }
                                    headers.put(argString.get(0), argString.get(1));
                                    LOGGER.info("107: Get New Header, {}->{}, Interceptor: {}", argString.get(0), argString.get(1), sootClass.getName());
                                }
                                else{
                                    HashMap<Integer, List<String>> args = new HashMap<>();
                                    int i = 0;
                                    for(Value arg: invokeExpr.getArgs()){
                                        if(localToString.containsKey(arg))
                                            args.put(i, localToString.get(arg));
                                        else if(arg instanceof Constant){
                                            Object obj = SimulateUtil.getConstant(arg);
                                            if(obj != null)
                                                args.put(i, List.of(obj.toString()));
                                        }
                                        i++;
                                    }
                                    List<String> result = MethodLocal.getStaticInvokeReturn(invokeExpr, args);
                                    if(!result.isEmpty())
                                        localToString.put(leftOp, result);
                                    else
                                        localToString.put(leftOp, List.of(method.getSignature()+MethodString.getContent(args)));
                                }
                            }
                        }
                    }
                    else if (unit instanceof InvokeStmt) {
                        InvokeExpr invokeExpr = ((InvokeStmt) unit).getInvokeExpr();
                        SootMethod method = ((InvokeStmt) unit).getInvokeExpr().getMethod();
                        String sig = method.getSignature();
                        if(invokeExpr instanceof InstanceInvokeExpr) {
                            Value base = ((InstanceInvokeExpr) invokeExpr).getBase();
                            if (sig.contains("Map: java.lang.Object put(java.lang.Object,java.lang.Object)>") ||
                                    sig.contains("JSONObject: org.json.JSONObject put(java.lang.String,java.lang.Object)>")) {
                                if(localToMap.containsKey(base)){
                                    List<List<String>> argString = new ArrayList<>();
                                    for(int i=0;i<2;i++){
                                        Value arg = invokeExpr.getArg(i);
                                        if(arg instanceof Constant){
                                            Object obj = SimulateUtil.getConstant(arg);
                                            if(obj != null)
                                                argString.add(List.of(obj.toString()));
                                        }
                                        else if(localToString.containsKey(arg)){
                                            argString.add(localToString.get(arg));
                                        }
                                        else
                                            argString.add(List.of("UNKNOWN"));
                                    }
                                    localToMap.get(base).put(argString.get(0), argString.get(1));
                                    LOGGER.info("New Put: {},{},{}", argString.get(0), argString.get(1),localToMap.get(base));
                                }

                            }
                            else if(method.getSignature().contains("okhttp3.Request$Builder: okhttp3.Request$Builder addHeader")){
                                List<List<String>> argString = new ArrayList<>();
                                for(int i=0;i<2;i++){
                                    Value arg = invokeExpr.getArg(i);
                                    if(arg instanceof Constant){
                                        Object obj = SimulateUtil.getConstant(arg);
                                        if(obj != null)
                                            argString.add(List.of(obj.toString()));
                                    }
                                    else if(localToString.containsKey(arg)){
                                        argString.add(localToString.get(arg));
                                    }
                                    else
                                        argString.add(List.of("UNKNOWN"));
                                }
                                headers.put(argString.get(0), argString.get(1));
                                LOGGER.info("179: Get New Header, {}->{}, Interceptor: {}", argString.get(0), argString.get(1), sootClass.getName());
                            }
                        }
                    }
                }

            }
        }
    }

    @Override
    public boolean equals(Object o) {
        if (this == o) return true;
        if (o == null || getClass() != o.getClass()) return false;
        Interceptor that = (Interceptor) o;
        return isNotCustomInterceptor == that.isNotCustomInterceptor && Objects.equals(sootClass, that.sootClass) && Objects.equals(currentMethod, that.currentMethod) && Objects.equals(headers, that.headers);
    }

    @Override
    public int hashCode() {
        return Objects.hash(sootClass, currentMethod, isNotCustomInterceptor, headers);
    }
}
