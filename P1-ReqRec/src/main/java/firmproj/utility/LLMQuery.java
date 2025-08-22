package firmproj.utility;

import firmproj.base.MethodLocal;
import firmproj.base.MethodParamInvoke;
import firmproj.base.MethodString;
import firmproj.client.AbstractHttpClient;
import org.apache.logging.log4j.LogManager;
import org.apache.logging.log4j.Logger;
import soot.*;
import soot.jimple.AssignStmt;
import soot.jimple.InvokeExpr;
import soot.jimple.InvokeStmt;
import soot.jimple.StaticInvokeExpr;

import java.util.*;

public class LLMQuery {
    private static final Logger LOGGER = LogManager.getLogger(LLMQuery.class);
    public static final HashSet<QueryJson> queries = new HashSet<>();
    private static final int RETRIEVE_STEP = 2;

    public static void generate(MethodParamInvoke methodParamInvoke){
        for(String invokeSig : methodParamInvoke.invokeMethodSig) {
            HashMap<String, HashMap<Integer, List<String>>> result = FirmwareRelated.matchInvokeSig(invokeSig);
            if(!result.isEmpty()) {
                for(Map.Entry<String, HashMap<Integer, List<String>>> entry : result.entrySet()) {
                    try{
                        SootMethod sootMethod = Scene.v().getMethod(entry.getKey());
                        if(MethodString.isStandardLibraryClass(sootMethod.getDeclaringClass()) || !sootMethod.isConcrete())
                            continue;
                        generate(sootMethod, entry.getValue());
                    }
                    catch (Exception ignore){}
                }
            }
        }
    }

    public static void generate(SootMethod sootMethod, HashMap<Integer, List<String>> paramValues){
        QueryJson queryJson = new QueryJson();
        SootClass sootClass = sootMethod.getDeclaringClass();
        if(MethodString.isStandardLibraryClass(sootClass)) return;

        List<List<String>> parameterValues = flattenParamValue(sootMethod, paramValues);
        queryJson.setTargetClass(sootClass.getName());
        queryJson.setTargetMethodSubsSig(sootMethod.getSubSignature());
        queryJson.setParameterValues(parameterValues);
        queryJson.targetMethod = sootMethod;

        List<String> methodSig = retrieveMethod(sootMethod);
        queryJson.setRelatedMethodsSig(methodSig);
        //queryJson.doGenerate();
        queries.add(queryJson);
    }

    public static QueryJson generateHttp(SootMethod sootMethod){
        QueryJson queryJson = new QueryJson();
        SootClass sootClass = sootMethod.getDeclaringClass();

        queryJson.setTargetClass(sootClass.getName());
        queryJson.setTargetMethodSubsSig(sootMethod.getSubSignature());
        queryJson.setParameterValues(new ArrayList<>());
        queryJson.targetMethod = sootMethod;

        List<String> methodSig = retrieveMethod(sootMethod);
        queryJson.setRelatedMethodsSig(methodSig);
        //queryJson.setHttp(true);
        //queryJson.doGenerate();

        queries.add(queryJson);
        return queryJson;
    }

    public static void checkAndGenerateJson(String str){
        for(QueryJson queryJson : queries){
            if(str.contains(queryJson.targetMethod.getSignature())){
                boolean succ = queryJson.doGenerate();
                if(succ)
                    LOGGER.info("Generate: {}", queryJson.getTargetMethodSubsSig());
            }
        }
    }


    public static List<List<String>> flattenParamValue(SootMethod sootMethod, HashMap<Integer, List<String>> paramValues){
        List<List<String>> result = new ArrayList<>();
        int num = sootMethod.getParameterCount();
        for(int i=0; i < num; i++){
            if(paramValues.containsKey(i)){
                result.add(paramValues.get(i));
            }
            else{
                result.add(new ArrayList<>());
            }
        }
        return result;
    }

    public static List<String> retrieveMethod(SootMethod method){
        return retrieveMethod(method, 0);
    }

    public static List<String> retrieveMethod(SootMethod method, int step){
        HashSet<String> result = new HashSet<>();
        if(step > RETRIEVE_STEP) return new ArrayList<>();
        SootClass sootClass = method.getDeclaringClass();
        if(MethodString.isStandardLibraryClass(sootClass) || !method.isConcrete()) return new ArrayList<>();
        Body body = method.retrieveActiveBody();
        for(Unit unit : body.getUnits()){
            if(unit instanceof AssignStmt){
                Value rightOp = ((AssignStmt) unit).getRightOp();
                if(rightOp instanceof InvokeExpr){
                    SootMethod invokeMethod = ((InvokeExpr) rightOp).getMethod();
                    String invokeMethodSig = invokeMethod.getSignature();
                    String sigLower = invokeMethodSig.toLowerCase();
                    SootClass declaringClz = invokeMethod.getDeclaringClass();
                    if(!MethodString.isStandardLibraryClass(declaringClz)){
                        if(sootClass.equals(declaringClz)){
                            result.add(invokeMethodSig);
                            result.addAll(retrieveMethod(invokeMethod, step + 1));
                        }
                        else if(rightOp instanceof StaticInvokeExpr){
                            if(sigLower.contains("log:") || sigLower.contains("log(") || sigLower.contains(".log.")) continue;
                            result.add(invokeMethodSig);
                            if(!sootClass.equals(declaringClz))
                                result.addAll(retrieveMethod(invokeMethod, step + 1));
                        }
                        else if(sigLower.contains("encode") || sigLower.contains("decode") || sigLower.contains("crypt") || sigLower.contains("util")) {
                            result.add(invokeMethodSig);
                            if (!sootClass.equals(declaringClz))
                                result.addAll(retrieveMethod(invokeMethod, step + 1));
                        }
                    }
                }
            }
            else if(unit instanceof InvokeStmt){

            }
        }
        return new ArrayList<>(result);
    }



}
