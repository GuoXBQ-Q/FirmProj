package firmproj.client;

import firmproj.base.MethodLocal;
import firmproj.base.MethodParamInvoke;
import firmproj.base.MethodString;
import firmproj.base.ValueContext;
import firmproj.graph.CallGraph;
import firmproj.graph.CallGraphNode;
import firmproj.objectSim.AbstractClz;
import firmproj.objectSim.SimulateUtil;
import firmproj.objectSim.UrlClz;
import firmproj.utility.FirmwareRelated;
import firmproj.utility.TimeoutTaskExecutor;
import org.apache.logging.log4j.LogManager;
import org.apache.logging.log4j.Logger;
import soot.*;
import soot.jimple.*;
import soot.util.Chain;

import java.util.*;
import java.util.stream.Collectors;

public class HttpClientFind {
    public static final HashMap<String, List<AbstractHttpClient>> findResult = new HashMap<>();

    private static final HashMap<String, Interceptor> allInterceptorClasses = new HashMap<>();

    private static final Logger LOGGER = LogManager.getLogger(HttpClientFind.class);

    public static final HashMap<SootField, List<AbstractHttpClient>> FieldToClientPoint = new HashMap<>();

    private static final HashSet<String> visitedMethod = new HashSet<>();

    public static final List<String> finalResult = new ArrayList<>();

    public static void findAllHttpClientBuildMethod(){
        Chain<SootClass> classes = Scene.v().getClasses();
        for (SootClass sootClass : classes) {
            String headString = sootClass.getName().split("\\.")[0];
            if (headString.contains("android") || headString.contains("kotlin") || headString.contains("java") || headString.contains("thingClips"))
                continue;
            for (SootMethod sootMethod : clone(sootClass.getMethods())) {
                if (!sootMethod.isConcrete())
                    continue;
                List<AbstractHttpClient> abstractHttpClients = findHttpClientBuildMethod(sootMethod);
                if(!abstractHttpClients.isEmpty()){
                    addValue(findResult, sootMethod.getSignature(), abstractHttpClients);
                }
            }
        }
        LOGGER.info("All HttpClient Result: {}",findResult.toString());
        //LOGGER.info("ALL Field with retrofitBuildPoint: {}", );
    }

    public static void findAllInterceptorClasses(){
        Chain<SootClass> classes = Scene.v().getClasses();
        for (SootClass sootClass : classes) {
            String headString = sootClass.getName().split("\\.")[0];
            String INTERCEPTOR = "okhttp3.Interceptor";
            if (headString.contains("android") || headString.contains("kotlin") || headString.contains("java") || headString.contains("thingClips"))
                continue;
            Chain<SootClass> interfaceCls = sootClass.getInterfaces();
            for(SootClass clz : interfaceCls){
                if(clz.getName().equals(INTERCEPTOR)) {
                    Interceptor interceptor = new Interceptor(sootClass);
                    interceptor.init();
                    allInterceptorClasses.put(sootClass.getName(), interceptor);
                    break;
                    //TODO start with okhttp3.* interceptor.
                }
            }
        }
        LOGGER.info("All Interceptor Classes Result : {}",allInterceptorClasses.toString());
        //LOGGER.info("ALL Field with retrofitBuildPoint: {}", );
    }

    public static void processAllClientParams(){
        // loop all client that need param.
        // get callNode, and the sootmethod Name / get all param.
        // if name or param is firm related, then output the result.
        HashMap<SootMethod, AbstractHttpClient> methodToClient = new HashMap<>();
        for(Map.Entry<String, List<AbstractHttpClient>> entry : findResult.entrySet()){
            for(AbstractHttpClient abstractHttpClient : entry.getValue()){
                if(abstractHttpClient.isNeedRequestContent()) {
                    SootMethod method = Scene.v().getMethod(entry.getKey());
//                    if (FirmwareRelated.isFirmRelated(method.getSubSignature())) {
//                        methodToClient.put(method, abstractHttpClient);
//                    }
//                    else{
                    if(method.getSignature().contains("onServerZone"))
                        LOGGER.info("Got");
                    List<CallGraphNode> nextNodes = CallGraph.findCaller(method.getSignature(), false);
                    for (CallGraphNode node : nextNodes) {
                        SootMethod nextMethod = node.getSootMethod();
                        if (FirmwareRelated.isFirmRelated(nextMethod.getSignature()) && !findResult.containsKey(nextMethod.getSignature())) {
                            AbstractHttpClient newClient = getAbstractHttpClient(abstractHttpClient);
                            newClient.setSootMethod(method);
                            methodToClient.put(nextMethod, newClient); // MethodLocal(nextMethod, newClient.sootMethod+params, 0);
                        }
                    }
//                    }
                }
                else{
                    if(abstractHttpClient instanceof CustomHttpClient){
                        if(FirmwareRelated.isFirmRelated(entry.getKey()) && abstractHttpClient.getResult().contains("http") && abstractHttpClient.getResult().contains("://")){
                            LOGGER.info("[SolvedClient]: {}\n{}", entry.getKey(), abstractHttpClient.getResult());
                            finalResult.add("[SolveClient]: "+ entry.getKey() + "\n" + abstractHttpClient.getResult());
                        }
                    }
                }
            }
        }

        for(Map.Entry<SootMethod, AbstractHttpClient> entry : methodToClient.entrySet()){
            HashMap<String, List<Integer>> nextInvokeParams = new HashMap<>();
            String sig = entry.getValue().getSootMethod().getSignature();
            nextInvokeParams.put(sig, MethodString.clone(entry.getValue().getParams()));
            MethodLocal methodLocal = new MethodLocal(entry.getKey(), nextInvokeParams,0);
            methodLocal.setGetResult(true);
            methodLocal.doAnalysis();
            if(methodLocal.getInterestingParamString().containsKey(sig)) {
                HashMap<Integer, List<String>> invokeResult = MethodString.clone(methodLocal.getInterestingParamString().get(sig));
                for(Integer integer : invokeResult.keySet()){
                    List<String> values = invokeResult.get(integer);
                    Iterator<String> iterator = values.iterator();
                    while(iterator.hasNext()){
                        String str = iterator.next();
                        if(str.startsWith("UNKNOWN") || str.isEmpty() || str.length() > 5000 || (values.size() > 10 && str.startsWith("<") && str.endsWith(">")))
                            iterator.remove();
                    }
                }
                entry.getValue().addParamValues(invokeResult);
                LOGGER.info("[SolveClient]: {}\n{}", entry.getKey().getSignature(), entry.getValue().getResult());
                finalResult.add("[SolveClient]: "+ entry.getKey().getSignature() + "\n" +  entry.getValue().getResult());
            }
        }

    }

    private static AbstractHttpClient getAbstractHttpClient(AbstractHttpClient abstractHttpClient) {
        AbstractHttpClient newClient;
        if(abstractHttpClient instanceof okHttpClient) {
            newClient = new okHttpClient((okHttpClient) abstractHttpClient);
        }
        else{
            newClient = new CustomHttpClient();
            if(abstractHttpClient.getRequestContent() != null)
                newClient.addRequestContent(abstractHttpClient.getRequestContent());
        }
        newClient.setSootMethod(abstractHttpClient.getSootMethod());
        newClient.setParams(abstractHttpClient.getParams());
        newClient.setNeedRequestContent(abstractHttpClient.isNeedRequestContent());
        return newClient;
    }


    public static List<AbstractHttpClient> findHttpClientBuildMethod(SootMethod sootMethod){
        String methodSig = sootMethod.getSignature();
        if(findResult.containsKey(methodSig)) return findResult.get(methodSig);

        List<AbstractHttpClient> result = new ArrayList<>();

        if(!sootMethod.isConcrete() || !sootMethod.getDeclaringClass().isApplicationClass() || MethodString.isStandardLibraryClass(sootMethod.getDeclaringClass())) return result;
        Body body = null;
        try {
            body = sootMethod.retrieveActiveBody();
        } catch (Exception e) {
            LOGGER.error("Could not retrieved the active body {} because {}", sootMethod, e.getLocalizedMessage());
        }
        if (body == null)
            return result;
        Type ret = sootMethod.getReturnType();

        if(visitedMethod.contains(methodSig)) return result;

        HashMap<Value, List<AbstractHttpClient>> localToPoint = new HashMap<>();
        HashMap<Value, List<List<String>>> localToArray = new HashMap<>();
        HashMap<Value, List<String>> localToString = new HashMap<>();
        HashMap<Value, Interceptor> localToInterceptor = new HashMap<>();
        HashMap<Value, HashMap<String, List<String>>> localToRequestBuilder = new HashMap<>();
        HashMap<String, HashMap<Integer, List<String>>> currentValues = new HashMap<>();
        HashMap<Value, List<Integer>> localFromParam = new HashMap<>();
        HashMap<Value, MethodParamInvoke> localFromParamInvoke = new HashMap<>();
        HashMap<Value, AbstractClz> localToClz = new HashMap<>();

        visitedMethod.add(methodSig);
        for (Unit unit : body.getUnits()) {
            Stmt stmt = (Stmt) unit;
            try {
                if (stmt instanceof IdentityStmt) {
                    IdentityStmt identityStmt = (IdentityStmt) stmt;
                    Value leftOperation = identityStmt.getLeftOp();
                    Value rightOperation = identityStmt.getRightOp();
                    if (leftOperation instanceof Local) {
                        if (rightOperation instanceof ParameterRef) {
                            ParameterRef ref = (ParameterRef) rightOperation;
                            Integer index = ref.getIndex();
                            addValue(localFromParam, leftOperation, index);
                        }
                    }
                } else if (stmt instanceof AssignStmt) {
                    Value leftOp = ((AssignStmt) stmt).getLeftOp();
                    Value rightOp = ((AssignStmt) stmt).getRightOp();
                    if(rightOp instanceof NewArrayExpr){
                        NewArrayExpr arrayExpr = (NewArrayExpr) rightOp;
                        Integer index = (Integer) SimulateUtil.getConstant(arrayExpr.getSize());
                        if(index != null && index > 0 && index < 100){
                            List<List<String>> arrayList = new ArrayList<>(index);
                            int i = 0;
                            while(i < index) {
                                arrayList.add(new ArrayList<>());
                                i++;
                            }
                            localToArray.put(leftOp, arrayList);
                            //LOGGER.info("297: New local Array: {}, {}, {},{}", stmt, leftOp, arrayList, index);
                        }
                        else{
                            localToArray.remove(leftOp);
                        }
                    }
                    else if (rightOp instanceof NewExpr) {
                        NewExpr newExpr = (NewExpr) rightOp;
                        String clsName = newExpr.getBaseType().getClassName();
                        SootClass cls = newExpr.getBaseType().getSootClass();
                        if (clsName.equals("okhttp3.OkHttpClient$Builder")) {
                            AbstractHttpClient point = new okHttpClient(sootMethod, unit);
                            addValue(localToPoint, leftOp, point);
                            result.add(point);
                            tryToAddResult(sootMethod, point);
                            LOGGER.info("94: Method: {} , new Client: {}", methodSig, point.toString());
                        } else if (clsName.equals("okhttp3.Request$Builder")) {
                            localToRequestBuilder.put(leftOp, new HashMap<>());
                        } else if (allInterceptorClasses.containsKey(clsName)) {
                            localToInterceptor.put(leftOp, allInterceptorClasses.get(clsName));
                        } else if(clsName.contains("java.net.URL")){
                            UrlClz urlClz = new UrlClz(cls, sootMethod);
                            localToClz.put(leftOp, urlClz);
                        } else if(MethodLocal.isCommonClz(clsName)){
                            AbstractClz abstractClz = MethodLocal.CreateCommonClz(cls, sootMethod);
                            if(abstractClz !=null)
                                localToClz.put(leftOp, abstractClz);
                        }
                    }
                    else if (rightOp instanceof InvokeExpr) {
                        SootMethod invokeMethod = ((InvokeExpr) rightOp).getMethod();
                        Type invokeRet = invokeMethod.getReturnType();
                        if (checkReturnType(invokeRet) && !invokeMethod.getDeclaringClass().getName().equals("okhttp3.OkHttpClient")) {
                            if (!invokeMethod.getSignature().contains("okhttp3.OkHttpClient$Builder: okhttp3.OkHttpClient build()")) {
                                List<AbstractHttpClient> points = findHttpClientBuildMethod(invokeMethod);
                                if (!points.isEmpty()) {
                                    result.addAll(points);
                                    for (AbstractHttpClient pt : points) {
                                        pt.setSootMethod(sootMethod);
                                        pt.setCreateUnit(unit);
                                        if (sootMethod.getReturnType().toString().equals("okhttp3.OkHttpClient")) {
                                            //TODO request body param trans.
                                            okHttpClient newClient = (okHttpClient) pt;
                                            newClient.setLocalValue(leftOp);
                                            addValue(localToPoint, leftOp, newClient);
                                            LOGGER.info("108: Method: {} , From Method: {}, new Client: {}", methodSig, invokeMethod.getSignature(), pt.toString());

                                        } else {
                                            addValue(localToPoint, leftOp, pt);
                                            LOGGER.info("114: Method: {} , From Method: {}, new Client: {}", methodSig, invokeMethod.getSignature(), pt.toString());
                                        }
                                    }
                                    tryToAddResult(sootMethod, points);
                                }
                            } else if (checkReturnInterceptor(invokeRet)) {
                                if (leftOp instanceof Local)
                                    localToInterceptor.put(leftOp, allInterceptorClasses.get(invokeRet.toString()));
                            }
                        }

                        Value base = rightOp instanceof InstanceInvokeExpr? ((InstanceInvokeExpr) rightOp).getBase() : null;
                        HashMap<Value, List<String>>  paramValueWithStrings = new HashMap<>();
                        HashMap<Integer, List<String>> paramIndexWithStrings = new HashMap<>();
                        int index = 0;
                        for(Value arg : ((InvokeExpr) rightOp).getArgs()){
                            if(localToClz.containsKey(arg)){
                                AbstractClz abstractClz1 = localToClz.get(arg);
                                abstractClz1.solve();
                                paramValueWithStrings.put(arg, new ArrayList<>(List.of(abstractClz1.toString())));

                                if(localFromParam.containsKey(arg) && base!=null){
                                    addValue(localFromParam, base, localFromParam.get(arg));
                                }
                            }
                            else if(localToArray.containsKey(arg)){
                                paramValueWithStrings.put(arg, new ArrayList<>(List.of(MethodString.getContent(localToArray.get(arg)))));
                            }
                            else if(arg instanceof Constant){
                                Object obj = SimulateUtil.getConstant(arg);
                                if(obj != null){
                                    String objString = obj.toString();
                                    paramValueWithStrings.put(arg, new ArrayList<>(List.of(objString)));
                                }
                            }
                            else if(localFromParamInvoke.containsKey(arg)){
                                paramValueWithStrings.put(arg, localFromParamInvoke.get(arg).invokeMethodSig);
                                if(!localFromParamInvoke.get(arg).param.isEmpty() && base!=null)
                                    addValue(localFromParam, base, localFromParamInvoke.get(arg).param);
                            }
                            else if(localToString.containsKey(arg) ){
                                paramValueWithStrings.put(arg, localToString.get(arg));
                            }
                            else if(localFromParam.containsKey(arg)){
                                if(base!=null)
                                    addValue(localFromParam, base, localFromParam.get(arg));
                                paramValueWithStrings.put(arg, new ArrayList<>(MethodString.paramListToString(localFromParam.get(arg))));
                            }
                            else{
                                paramValueWithStrings.put(arg, new ArrayList<>(List.of("UNKOWN")));
                            }
                            paramIndexWithStrings.put(index, paramValueWithStrings.get(arg));
                            index++;
                        }

                        if (rightOp instanceof InstanceInvokeExpr) {
                            InvokeExpr invokeExpr = (InstanceInvokeExpr) rightOp;
                            String sig = ((InstanceInvokeExpr) rightOp).getMethod().getSignature();
                            if(localToClz.containsKey(base)){
                                AbstractClz abstractClz = localToClz.get(base);

                                ValueContext valueContext = new ValueContext(sootMethod, unit, paramValueWithStrings);
                                abstractClz.addValueContexts(valueContext);
                                abstractClz.solve();
                                localToClz.put(leftOp, localToClz.get(base));
                                localFromParam.remove(leftOp);

                                if(abstractClz instanceof UrlClz) {
                                    if(localFromParam.containsKey(base)) {
                                        addValue(((UrlClz) abstractClz).getClientResult().params, localFromParam.get(base));
                                        ((UrlClz) abstractClz).getClientResult().setNeedRequestContent(true);
                                    }
                                    if(((UrlClz) abstractClz).isUrlClient()) {
                                        addValue(localToPoint, leftOp, ((UrlClz) abstractClz).getClientResult());
                                        addValue(result, ((UrlClz) abstractClz).getClientResult());
                                    }
                                }
                            }
                            else if (localToPoint.containsKey(base)) {
                                List<AbstractHttpClient> localPoints = localToPoint.get(base);
                                if (sig.contains("okhttp3.OkHttpClient$Builder: okhttp3.OkHttpClient$Builder addInterceptor") ||
                                        sig.contains("okhttp3.OkHttpClient$Builder: okhttp3.OkHttpClient$Builder addNetworkInterceptor")) {
                                    Value arg = invokeExpr.getArg(0);
                                    if (localToInterceptor.containsKey(arg)) {
                                        for (AbstractHttpClient client : localPoints) {
                                            okHttpClient okHttpClient = (firmproj.client.okHttpClient) client;
                                            okHttpClient.setInterceptors(localToInterceptor.get(arg));
                                        }
                                    }
                                } else if (sig.contains("okhttp3.OkHttpClient$Builder: okhttp3.OkHttpClient build()")) {
                                    localToPoint.put(leftOp, localToPoint.remove(base));
                                    for (AbstractHttpClient client : localToPoint.get(leftOp)) {
                                        okHttpClient newClient = (okHttpClient) client;
                                        newClient.setLocalValue(leftOp);
                                    }
                                } else if (sig.contains("okhttp3.OkHttpClient: okhttp3.Call newCall")) {
                                    localToPoint.put(leftOp, localToPoint.remove(base));
                                    Value arg0 = invokeExpr.getArg(0);
                                    HashMap<String, List<String>> newCallParam = new HashMap<>();
                                    if (localToRequestBuilder.containsKey(arg0)) {
                                        newCallParam.putAll(localToRequestBuilder.get(arg0));
                                    }else if (localFromParam.containsKey(arg0)) {
                                        newCallParam.put("Request", MethodString.paramListToString(localFromParam.get(arg0)));
                                    }
                                    for (AbstractHttpClient client : localToPoint.get(leftOp)) {
                                        okHttpClient okHttpClient = (firmproj.client.okHttpClient) client;
                                        if(localFromParam.containsKey(arg0)) {
                                            okHttpClient.setNeedRequestContent(true);
                                            addValue(okHttpClient.params, localFromParam.get(arg0));
                                        }
                                        okHttpClient.requestContentFromParams.putAll(newCallParam);
                                    }
                                } else if (sig.contains("com.github.kittinunf.fuel.core.Request: com.github.kittinunf.fuel.core.Request header(java.util.Map)")) {
                                    Value arg = invokeExpr.getArg(0);
                                    List<Integer> params = new ArrayList<>();
                                    if(localFromParamInvoke.containsKey(arg)){
                                        params = new ArrayList<>(localFromParamInvoke.get(arg).param);
                                    }
                                    else if(localFromParam.containsKey(arg)){
                                        params = new ArrayList<>(localFromParam.get(arg));
                                    }
                                    for (AbstractHttpClient abstractHttpClient : localPoints) {
                                        CustomHttpClient customHttpClient = (CustomHttpClient) abstractHttpClient;
                                        customHttpClient.requestContentFromParams.put("headers", paramValueWithStrings.get(arg));
                                        if(!params.isEmpty()) {
                                            addValue(customHttpClient.params, params);
                                            customHttpClient.setNeedRequestContent(true);
                                        }
                                    }


                                }
                            } else if (localToRequestBuilder.containsKey(base)) {
                                if (sig.contains("okhttp3.Request$Builder: okhttp3.Request$Builder url")) {
                                    Value urlParam = invokeExpr.getArg(0);
                                    if (urlParam instanceof Constant) {
                                        Object obj = SimulateUtil.getConstant(urlParam);
                                        if (obj != null) {
                                            localToRequestBuilder.get(base).put("url", List.of(obj.toString()));
                                        }
                                    } else if(localToClz.containsKey(urlParam)){
                                        AbstractClz abstractClz1 = localToClz.get(urlParam);
                                        abstractClz1.solve();
                                        if(localFromParam.containsKey(urlParam)){
                                            addValue(localFromParam, base, localFromParam.get(urlParam));
                                        }
                                    }
                                    else if (localFromParam.containsKey(urlParam)) {
                                        localToRequestBuilder.get(base).put("url", MethodString.paramListToString(localFromParam.get(urlParam))); //todo replace
                                        addValue(localFromParam, base, localFromParam.get(urlParam));
                                    } else if (urlParam instanceof Local) {
                                        if (!currentValues.containsKey(sig)) {
                                            HashMap<String, List<Integer>> interestingInvoke = new HashMap<>();
                                            interestingInvoke.put(sig, new ArrayList<>(List.of(0)));
                                            TimeoutTaskExecutor timeoutTaskExecutor = new TimeoutTaskExecutor(10);
                                            timeoutTaskExecutor.executeWithTimeout(() -> {
                                            MethodLocal methodLocal = new MethodLocal(sootMethod, interestingInvoke, 0);
                                            methodLocal.setGetResult(true);
                                            methodLocal.doAnalysis();
                                            if (methodLocal.getLocalFromParams().containsKey(invokeExpr.getArg(0))) {
                                                localToRequestBuilder.get(base).put("url", MethodString.paramListToString(methodLocal.getLocalFromParams().get(invokeExpr.getArg(0))));
                                            } else {
                                                currentValues.putAll(methodLocal.getInterestingParamString());
                                            }});
                                            timeoutTaskExecutor.shutdown();
                                        }
                                        if (currentValues.containsKey(sig)) {
                                            List<String> urls = currentValues.get(sig).get(0);
                                            if (urls != null) {
                                                localToRequestBuilder.get(base).put("url", urls);
                                            }
                                        }
                                    }
                                } else if (sig.contains("okhttp3.Request$Builder: okhttp3.Request$Builder post")) {
                                    Value postParam = invokeExpr.getArg(0);
                                    if (postParam instanceof Constant) {
                                        Object obj = SimulateUtil.getConstant(postParam);
                                        if (obj != null) {
                                            localToRequestBuilder.get(base).put("body", List.of(obj.toString()));
                                        }
                                    } else if (localToString.containsKey(postParam)) {
                                        localToRequestBuilder.get(base).put("body", localToString.get(postParam));
                                        if(localFromParam.containsKey(postParam)){
                                            addValue(localFromParam, base, localFromParam.get(postParam));
                                        }
                                    }else if (localFromParam.containsKey(postParam)) {
                                        localToRequestBuilder.get(base).put("body", MethodString.paramListToString(localFromParam.get(postParam)));
                                        addValue(localFromParam, base, localFromParam.get(postParam));
                                    }
                                } else if (sig.contains("okhttp3.Request$Builder: okhttp3.Request build()")) {
                                    localToRequestBuilder.put(leftOp, localToRequestBuilder.remove(base));
                                }
                            }
                            else if(MethodString.methodReturnParamInvoke.containsKey(invokeMethod)){
                                MethodParamInvoke methodParamInvoke = new MethodParamInvoke(MethodString.methodReturnParamInvoke.get(invokeMethod));
                                if(!methodParamInvoke.paramValue.isEmpty() || methodParamInvoke.param.isEmpty()) {
                                    localToString.put(leftOp, methodParamInvoke.invokeMethodSig);
                                    localFromParam.remove(leftOp);
                                    localToClz.remove(leftOp);
                                    localFromParamInvoke.remove(leftOp);
                                    localToPoint.remove(leftOp);
                                }
                                else {
                                    List<Integer> params = new ArrayList<>(methodParamInvoke.param);
                                    methodParamInvoke.param.clear();
                                    methodParamInvoke.sootMethod = sootMethod;
                                    HashMap<Integer, List<String>> paramValues = new HashMap<>();
                                    for (int i : params) {
                                        Value arg = invokeExpr.getArg(i);
                                        if (paramValueWithStrings.containsKey(arg)) { //todo just localtostring or constant
                                            paramValues.put(i, paramValueWithStrings.get(arg));
                                        }
                                        if (localFromParam.containsKey(arg)) {
                                            paramValues.remove(i);
                                            addValue(methodParamInvoke.param, localFromParam.get(arg));
                                            methodParamInvoke.invokeMethodSig.replaceAll(s -> s.replace("$["+i+"]", MethodString.getContent(MethodString.paramListToString(localFromParam.get(arg)))));
                                        }
                                        else if(localFromParamInvoke.containsKey(arg)){
                                            paramValues.remove(i);
                                            addValue(methodParamInvoke.param, localFromParamInvoke.get(arg).param);
                                            methodParamInvoke.invokeMethodSig.replaceAll(s -> s.replace("$["+i+"]", MethodString.getContent(localFromParamInvoke.get(arg).invokeMethodSig)));
                                        }
                                    }
                                    if (!paramValues.isEmpty()) {
                                        methodParamInvoke.addParamValue(paramValues);
                                        methodParamInvoke.solve();
                                    }
                                    if(methodParamInvoke.param.isEmpty()){
                                        localToString.put(leftOp,  new ArrayList<>(List.of(MethodString.getContent(methodParamInvoke.invokeMethodSig) + MethodString.getContent(methodParamInvoke.paramValue))));
                                    }
                                    else {
                                        localFromParamInvoke.put(leftOp, methodParamInvoke);
                                    }
                                    localFromParam.remove(leftOp);
                                    localToClz.remove(leftOp);
                                    localFromParamInvoke.remove(leftOp);
                                    localToPoint.remove(leftOp);
                                }
                            }
                            else if(MethodString.methodToFieldString.containsKey(invokeMethod) || MethodString.getMethodToString().containsKey(invokeMethod)) {
                                if (MethodString.methodToString.containsKey(invokeMethod)) {
                                    localToString.put(leftOp, MethodString.getMethodToString().get(invokeMethod));
                                } else if (MethodString.methodToFieldString.containsKey(invokeMethod)) {
                                    localToString.put(leftOp, new ArrayList<>(List.of(MethodString.methodToFieldString.get(invokeMethod))));
                                }
                                localFromParam.remove(leftOp);
                                localFromParamInvoke.remove(leftOp);
                                localToClz.remove(leftOp);
                            }
                            else {
                                if (localFromParamInvoke.containsKey(base)) {
                                    localFromParamInvoke.put(leftOp, new MethodParamInvoke(localFromParamInvoke.get(base)));
                                    localFromParamInvoke.get(leftOp).addMethodInvoke(invokeMethod.getSignature());
                                    if(localFromParam.containsKey(base))
                                        addValue(localFromParamInvoke.get(base).param, localFromParam.get(base));
                                } else if (localFromParam.containsKey(base)) {
                                    localFromParamInvoke.put(leftOp, new MethodParamInvoke(sootMethod, localFromParam.get(base), invokeMethod.getSignature() + paramIndexWithStrings));
                                } else {
                                    MethodParamInvoke methodParamInvoke = new MethodParamInvoke(sootMethod, new ArrayList<>(), invokeMethod.getSignature() + paramIndexWithStrings);
                                    localFromParamInvoke.put(leftOp, methodParamInvoke);
                                }
                                localToClz.remove(leftOp);
                                localToArray.remove(leftOp);
                                localToString.remove(leftOp);
                                localFromParam.remove(leftOp);
                            }
                            if(localFromParam.containsKey(base) && !((Local)base).getName().equals("r0")){
                                addValue(localFromParam, leftOp, localFromParam.get(base));
                            }

                        } else if (rightOp instanceof StaticInvokeExpr) {
                            StaticInvokeExpr staticInvokeExpr = (StaticInvokeExpr) rightOp;
                            String invokeSig = staticInvokeExpr.getMethod().getSignature();
                            if (invokeSig.contains("okhttp3.MediaType: okhttp3.MediaType parse")) {
                                localToString.put(leftOp, new ArrayList<>());
                                Value arg0 = staticInvokeExpr.getArg(0);
                                if (arg0 instanceof Constant) {
                                    Object obj = SimulateUtil.getConstant(staticInvokeExpr.getArg(0));
                                    if (obj != null) {
                                        localToString.get(leftOp).add(obj.toString());
                                    }
                                } else if (localFromParam.containsKey(arg0)) {
                                    localToString.get(leftOp).add(MethodString.getContent(MethodString.paramListToString(localFromParam.get(arg0))));
                                }
                            } else if (invokeSig.contains("okhttp3.RequestBody: okhttp3.RequestBody create")) {
                                localToString.put(leftOp, new ArrayList<>());
                                for (Value value : staticInvokeExpr.getArgs()) {//todo replace with parameterValues
                                    if (value instanceof Constant) {
                                        Object obj = SimulateUtil.getConstant(value);
                                        if (obj != null) {
                                            localToString.get(leftOp).add(obj.toString());
                                        }
                                    } else if (localToString.containsKey(value)) {
                                        localToString.get(leftOp).addAll(localToString.get(value));
                                    }else if (localFromParam.containsKey(value)) {
                                        localToString.get(leftOp).add(MethodString.getContent(MethodString.paramListToString(localFromParam.get(value))));
                                        localFromParam.remove(leftOp);
                                        addValue(localFromParam, leftOp, localFromParam.get(value));
                                    }
                                }
                            } else if (invokeSig.contains("com.github.kittinunf.fuel.FuelKt: com.github.kittinunf.fuel.core.Request httpGet(java.lang.String,java.util.List)") ||
                                    invokeSig.contains("com.github.kittinunf.fuel.FuelKt: com.github.kittinunf.fuel.core.Request httpPost(java.lang.String,java.util.List)")) {
                                CustomHttpClient customHttpClient = new CustomHttpClient(invokeMethod, unit);
                                addValue(localToPoint, leftOp, customHttpClient);
                                if(MethodString.fieldToString.containsKey("<com.github.kittinunf.fuel.core.FuelManager: java.lang.String basePath>")
                                || MethodString.fieldToString.containsKey("<com.github.kittinunf.fuel.core.FuelManager: java.util.Map baseHeaders>")
                                || MethodString.fieldToString.containsKey("<com.github.kittinunf.fuel.core.FuelManager: java.util.List baseParams>")){
                                    HashMap<String, List<String>> requestContentFromParams = new HashMap<>();
                                    List<String> tmp = MethodString.fieldToString.get("<com.github.kittinunf.fuel.core.FuelManager: java.lang.String basePath>");
                                    if(tmp != null)
                                        requestContentFromParams.put("basePath", clone(tmp));
                                    tmp = MethodString.fieldToString.get("<com.github.kittinunf.fuel.core.FuelManager: java.util.Map baseHeaders>");
                                    if(tmp != null)
                                        requestContentFromParams.put("headers", clone(tmp));
                                    tmp = MethodString.fieldToString.get("<com.github.kittinunf.fuel.core.FuelManager: java.util.List baseParams>");
                                    if(tmp != null)
                                        requestContentFromParams.put("baseParams", clone(tmp));
                                    if(!requestContentFromParams.isEmpty())
                                        customHttpClient.addRequestContent(requestContentFromParams);
                                }

                                result.add(customHttpClient);
                                for(Value arg : staticInvokeExpr.getArgs()){
                                    if(localFromParam.containsKey(arg)){
                                        addValue(customHttpClient.params, localFromParam.get(arg));
                                        customHttpClient.setNeedRequestContent(true);
                                    }else if(localFromParamInvoke.containsKey(arg)){
                                        if(!localFromParamInvoke.get(arg).param.isEmpty()) {
                                            addValue(customHttpClient.params, localFromParamInvoke.get(arg).param);
                                            customHttpClient.setNeedRequestContent(true);
                                        }
                                    }
                                    if(staticInvokeExpr.getArg(0).equals(arg))
                                        customHttpClient.requestContentFromParams.put("url", paramValueWithStrings.get(arg));
                                    else{
                                        customHttpClient.requestContentFromParams.put("body", paramValueWithStrings.get(arg));
                                    }
                                }
                            } else {
                                List<Integer> argFromParams = new ArrayList<>();
                                for(Value arg : staticInvokeExpr.getArgs()) {
                                    if (localFromParam.containsKey(arg)) {
                                        addValue(argFromParams,localFromParam.get(arg));
                                    }
                                    else if(localFromParamInvoke.containsKey(arg)){
                                        addValue(argFromParams, localFromParamInvoke.get(arg).param);
                                    }
                                }
                                List<String> staticInvokeReturn = MethodLocal.getStaticInvokeReturn(staticInvokeExpr, paramIndexWithStrings);
                                if(!staticInvokeReturn.isEmpty()){
                                    MethodParamInvoke methodParamInvoke = new MethodParamInvoke(sootMethod, argFromParams, staticInvokeReturn);
                                    localFromParamInvoke.put(leftOp, methodParamInvoke);
                                }
                                else{
                                    localFromParamInvoke.put(leftOp, new MethodParamInvoke(sootMethod, argFromParams, invokeMethod.getSignature()));
                                }
                                localToClz.remove(leftOp);
                                localToArray.remove(leftOp);
                                localToString.remove(leftOp);
                                localFromParam.remove(leftOp);
                            }
                        }
                    } else if (rightOp instanceof CastExpr) {
                        Value op = ((CastExpr) rightOp).getOp();
                        localToClz.remove(leftOp);
                        localToString.remove(leftOp);
                        localToPoint.remove(leftOp);
                        if(localFromParam.containsKey(op)){
                            localFromParam.put(leftOp, localFromParam.get(op));
                        }
                        if (localToPoint.containsKey(op)) {
                            if (leftOp instanceof Local)
                                localToPoint.put(leftOp, localToPoint.get(op));
                        } else if (localToInterceptor.containsKey(op)) {
                            localToInterceptor.put(leftOp, localToInterceptor.get(op));
                        }
                        if (localToClz.containsKey(op)) {
                            localToClz.put(leftOp, localToClz.get(op));
                        }
                    } else if (rightOp instanceof Local && localToPoint.containsKey(rightOp)) {
                        if (leftOp instanceof FieldRef) {
                            FieldToClientPoint.put(((FieldRef) leftOp).getField(), localToPoint.get(rightOp));
                        }
                    } else if (rightOp instanceof Local && leftOp instanceof ArrayRef) {
                        ArrayRef arrayRef = (ArrayRef) leftOp;
                        Integer arrayIndex = (Integer) SimulateUtil.getConstant(arrayRef.getIndex());
                        Value base = arrayRef.getBase();
                        if(localToArray.containsKey(base)) {
                            if (arrayIndex != null) {
                                try {List<String> arrayValue = localToArray.get(base).get(arrayIndex);

                                    if (rightOp instanceof Constant) {
                                        Object obj = SimulateUtil.getConstant(rightOp);
                                        if(obj != null)
                                            arrayValue.add(obj.toString());
                                    }
                                    else if(localToString.containsKey(rightOp)){
                                        addValue(arrayValue, localToString.get(rightOp));
                                    }
                                    else if(localToClz.containsKey(rightOp)){
                                        AbstractClz abstractClz = localToClz.get(rightOp);
                                        abstractClz.solve();
                                        if(abstractClz.isSolved())
                                            arrayValue.add(abstractClz.toString());
                                        if(localFromParam.containsKey(rightOp)){
                                            addValue(localFromParam, base, localFromParam.get(rightOp));
                                        }
                                    }
                                    else if (localFromParam.containsKey(rightOp)) {
                                        arrayValue.clear();
                                        arrayValue.addAll(MethodString.paramListToString(localFromParam.get(rightOp)));
                                        addValue(localFromParam, base, localFromParam.get(rightOp));
                                    }
                                    else if (localFromParamInvoke.containsKey(rightOp)){
                                        MethodParamInvoke methodParamInvoke = localFromParamInvoke.get(rightOp);
                                        addValue(arrayValue, methodParamInvoke.invokeMethodSig);
                                        if(!methodParamInvoke.param.isEmpty())
                                            addValue(localFromParam, base, methodParamInvoke.param);
                                    }
                                }
                                catch (Exception e){LOGGER.error(e);}
                            }
                        }
                        else if(localFromParamInvoke.containsKey(rightOp)){
                            localFromParamInvoke.put(leftOp, new MethodParamInvoke(localFromParamInvoke.get(rightOp)));
                        }

                    } else if (localToClz.containsKey(rightOp)) {
                        localToClz.put(leftOp, localToClz.get(rightOp));
                    } else if (rightOp instanceof FieldRef) {
                        SootField field = ((FieldRef) rightOp).getField();
                        if(MethodString.fieldToString.containsKey(field.toString())){
                            localToString.put(leftOp, MethodString.fieldToString.get(field.toString()));
                            localFromParam.remove(leftOp);
                            localToClz.remove(leftOp);
                            localToPoint.remove(leftOp);
                        }
                        if (FieldToClientPoint.containsKey(field)) {
                            if (leftOp instanceof Local) {
                                localToPoint.put(leftOp, FieldToClientPoint.get(((FieldRef) rightOp).getField()));
                                tryToAddResult(sootMethod, localToPoint.get(leftOp));
                            }
                        } else if (field.getType() instanceof RefType) {
                            String clsName = ((RefType) field.getType()).getClassName();
                            if (allInterceptorClasses.containsKey(clsName)) {
                                localToInterceptor.put(leftOp, allInterceptorClasses.get(clsName));
                            }
                        }
                    } else {
                        localToArray.remove(leftOp);
                        localToClz.remove(leftOp);
                    }
                }else if(stmt instanceof InvokeStmt){
                    if(stmt.getInvokeExpr() instanceof InstanceInvokeExpr) {
                        Value base = ((InstanceInvokeExpr) stmt.getInvokeExpr()).getBase();
                        InvokeExpr invokeExpr = stmt.getInvokeExpr();
                        if(localToClz.containsKey(base)){
                            AbstractClz abstractClz = localToClz.get(base);
                            HashMap<Value, List<String>>  paramValueWithStrings = new HashMap<>();

                            for(Value arg : invokeExpr.getArgs()){
                                if(localToClz.containsKey(arg)){
                                    AbstractClz abstractClz1 = localToClz.get(arg);
                                    abstractClz1.solve();
                                    if(abstractClz1.isSolved()){
                                        paramValueWithStrings.put(arg, new ArrayList<>(List.of(abstractClz1.toString())));
                                    }
                                    if(localFromParam.containsKey(arg)){
                                        addValue(localFromParam, base, localFromParam.get(arg));
                                    }
                                }
                                else if(arg instanceof Constant){
                                    Object obj = SimulateUtil.getConstant(arg);
                                    if(obj != null){
                                        String objString = obj.toString();
                                        paramValueWithStrings.put(arg, new ArrayList<>(List.of(objString)));
                                    }
                                }
                                else if(localToString.containsKey(arg) ){
                                    paramValueWithStrings.put(arg, localToString.get(arg));
                                }
                                else if(localFromParam.containsKey(arg)){
                                    addValue(localFromParam, base, localFromParam.get(arg));
                                    paramValueWithStrings.put(arg, MethodString.paramListToString(localFromParam.get(arg)));
                                }

                            }
                            ValueContext valueContext = new ValueContext(sootMethod, unit, paramValueWithStrings);
                            abstractClz.addValueContexts(valueContext);
                            abstractClz.solve();

                            if(abstractClz instanceof UrlClz && localFromParam.containsKey(base))
                                addValue(((UrlClz)abstractClz).getClientResult().params, localFromParam.get(base));
                        }
                    }
                }
            }catch (Throwable ignore){
            }
        }

        return result.stream().distinct().collect(Collectors.toList());
    }

    private static void tryToAddResult(SootMethod sootMethod, AbstractHttpClient point){
        addValue(findResult, sootMethod.getSignature(), point);
    }

    private static void tryToAddResult(SootMethod sootMethod, List<AbstractHttpClient> points){
        for (AbstractHttpClient point : points)
            addValue(findResult, sootMethod.getSignature(), point);
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

    public static <K, V> void addValue(Map<K, List<V>> map, Map<K, List<V>> map1) {
        for(K key : map1.keySet()){
            addValue(map, key, map1.get(key));
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

    public static HashMap<Integer, String> checkArgType(InvokeExpr invokeExpr){
        List<Value> args = invokeExpr.getArgs();
        HashMap<Integer, String> result = new HashMap<>();
        String clsString = "";
        int i = 0;
        for(Value arg : args){
            if(arg instanceof ClassConstant){
                clsString = (String)SimulateUtil.getConstant(arg);
                result.put(i,clsString);
            }
            i++;
        }
        return result;
    }

    public static boolean checkReturnType(Type v){
        String typeClz = v.toString();
        for(SootClass clz : RetrofitBuildFind.RetrofitClassesWithMethods.keySet()){
            if(typeClz.equals(clz.getName()))
                return true;
        }
        return typeClz.equals("java.lang.Object") || typeClz.contains("okhttp3.OkHttpClient") || typeClz.contains("retrofit2.Retrofit");
    }

    public static boolean checkReturnInterceptor(Type v){
        String typeClz = v.toString();
        for(String clz : allInterceptorClasses.keySet()){
            if(typeClz.equals(clz))
                return true;
        }
        return false;
    }


    public static <T> List<T> clone(List<T> ls) {
        return new ArrayList<T>(ls);
    }

}
