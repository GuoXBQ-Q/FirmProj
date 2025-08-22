package firmproj.client;

import firmproj.base.MethodLocal;
import firmproj.base.MethodString;
import firmproj.base.RetrofitPoint;
import firmproj.objectSim.SimulateUtil;
import org.apache.logging.log4j.LogManager;
import org.apache.logging.log4j.Logger;
import soot.*;
import soot.jimple.*;
import soot.util.Chain;

import java.util.*;

public class RetrofitBuildFind {

    public static final HashMap<String,List<RetrofitBuildPoint>> findResult = new HashMap<>(); //Return Api method's Name To RetrofitBuildPoint
    public static final HashMap<String,List<RetrofitBuildPoint>> RetrofitClassToBuildPoint = new HashMap<>(); //Retrofit create api -> class.
    public static final HashMap<SootClass, List<RetrofitPoint>> RetrofitClassesWithMethods = new HashMap<>(); //Retrofit Interface Class -> all the class Retrofit Point.
    public static final HashMap<SootField, List<RetrofitBuildPoint>> FieldToBuildPoint = new HashMap<>(); //SootField related to the RetrofitBuildPoint
    public static final HashMap<String, AbstractFactory> allFactoryClass = new HashMap<>();

    private static final Logger LOGGER = LogManager.getLogger(RetrofitBuildFind.class);
    private static final HashSet<String> visitedMethod = new HashSet<>();

    public static void findAllRetrofitBuildMethod(){
        Chain<SootClass> classes = Scene.v().getClasses();
        for (SootClass sootClass : classes) {
            String headString = sootClass.getName().split("\\.")[0];
            if (headString.contains("android") || headString.contains("kotlin") || headString.contains("java") || headString.contains("thingClips"))
                continue;
            //LOGGER.info(sootClass.getName().toString());
            for (SootMethod sootMethod : clone(sootClass.getMethods())) {
                if (!sootMethod.isConcrete())
                    continue;
                List<RetrofitBuildPoint> retrofitBuildPoints = findRetrofitBuildMethod(sootMethod);
                if(!retrofitBuildPoints.isEmpty())
                    addValue(findResult, sootMethod.getSignature(), retrofitBuildPoints);
            }
        }
        LOGGER.info("All retrofitResult: {}",findResult.toString());
        LOGGER.info("ALL Field with retrofitBuildPoint: {}", FieldToBuildPoint);
    }

    public static void findAllFactoryClasses(){
        Chain<SootClass> classes = Scene.v().getClasses();
        for (SootClass sootClass : classes) {
            if(MethodString.isStandardLibraryClass(sootClass)) continue;
            String headString = sootClass.getName().split("\\.")[0];
            String CONVERTER_FACTORY = "retrofit2.Converter$Factory";
            String CALL_FACTORY = "okhttp3.Call$Factory";
            if (headString.contains("android") || headString.contains("kotlin") || headString.contains("java") || headString.contains("thingClips"))
                continue;
            SootClass superClz = sootClass.getSuperclass();
            Chain<SootClass> interfaces = sootClass.getInterfaces();
            if(superClz != null) {
                if (superClz.getName().equals(CONVERTER_FACTORY)) {
                    ConverterFactory converterFactory = new ConverterFactory(sootClass);
                    converterFactory.init();
                    allFactoryClass.put(sootClass.getName(), converterFactory);
                }
            }
            if(!interfaces.isEmpty()) {
                for (SootClass interfaceClz : interfaces) {
                    if (interfaceClz.getName().equals(CALL_FACTORY)) {
                        CallFactory callFactory = new CallFactory(sootClass);
                        callFactory.init();
                        allFactoryClass.put(sootClass.getName(), callFactory);
                    }
                }
            }
        }
        LOGGER.info("All Factory Classes Result : {}",allFactoryClass.toString());

        //LOGGER.info("ALL Field with retrofitBuildPoint: {}", );
    }

    public static void processPointParams(List<RetrofitPoint> firmRelatedPoints){
        //TODO, solve
        //loop all points, get call node and find params.
        // merge the annotations , params, buildPoint's info. output result.
    }

    private static final HashSet<String> unsolvedMethod = new HashSet<>();

    public static List<RetrofitBuildPoint> findRetrofitBuildMethod(SootMethod sootMethod){
        String methodSig = sootMethod.getSignature();
        if(methodSig.contains("com.liesheng.hayloufun.repository.a: com.liesheng.hayloufun.repository.ApiService a()"))
            LOGGER.info("got");
        if(methodSig.contains("cn.hle.lhzm.ui.activity.user.ChangePhoneEmailActivity:") && methodSig.contains("init"))
            LOGGER.info("got");
        if(methodSig.contains("com.liesheng.module_device.setting.viewmodel.OtaCheckViewModel: void <init>(android.app.Application)"))
            LOGGER.info("got");
        if(findResult.containsKey(methodSig)) return findResult.get(methodSig);

        List<RetrofitBuildPoint> result = new ArrayList<>();

        if(visitedMethod.contains(methodSig)) return result;

        if(!sootMethod.isConcrete() || !sootMethod.getDeclaringClass().isApplicationClass() || MethodString.isStandardLibraryClass(sootMethod.getDeclaringClass())) return result;
        Body body = null;
        try {
            body = sootMethod.retrieveActiveBody();
        } catch (Exception e) {
            //LOGGER.error("Could not retrieved the active body {} because {}", sootMethod, e.getLocalizedMessage());
        }
        if (body == null)
            return result;
        Type ret = sootMethod.getReturnType();

        HashMap<Value, List<RetrofitBuildPoint>> localToPoint = new HashMap<>();
        HashMap<String, HashMap<Integer, List<String>>> currentValues = new HashMap<>();
        HashMap<Value, List<AbstractHttpClient>> localToClient = new HashMap<>();
        HashMap<Value, AbstractFactory> localToFactory = new HashMap<>();

        if(!checkReturnType(ret)) return result;
        visitedMethod.add(methodSig);

        for (Unit unit : body.getUnits()) {
            Stmt stmt = (Stmt) unit;
            if(stmt instanceof AssignStmt){
                Value leftOp = ((AssignStmt) stmt).getLeftOp();
                Value rightOp = ((AssignStmt) stmt).getRightOp();
                if(rightOp instanceof NewExpr){
                    NewExpr newExpr = (NewExpr) rightOp;
                    String clsName = newExpr.getBaseType().getClassName();
                    SootClass newClz = newExpr.getBaseType().getSootClass();
                    if(clsName.equals("retrofit2.Retrofit$Builder") || clsName.toLowerCase().contains("retrofitbuilder")){
                        RetrofitBuildPoint point = new RetrofitBuildPoint(sootMethod, unit);
                        addValue(localToPoint, leftOp, point);
                        result.add(point);
                        tryToAddResult(sootMethod, point);
                        LOGGER.info("72: Method: {} , new Point: {}", methodSig, point.toString());
                    }
                    else if(allFactoryClass.containsKey(clsName)){
                        localToFactory.put(leftOp, allFactoryClass.get(clsName));
                    }
                    else if(checkIfKotlinFunction0(newClz)){
                        SootMethod invokeM = newClz.getMethod("java.lang.Object invoke()");
                        if(invokeM != null){
                            findRetrofitBuildMethod(invokeM);
                            if(findResult.containsKey(invokeM.getSignature())){
                                if (leftOp instanceof Local) {
                                    localToPoint.put(leftOp,findResult.get(invokeM.getSignature()));
                                    addValue(result, findResult.get(invokeM.getSignature()));
                                }
                            }
                        }
                    }
                }
                if(rightOp instanceof InvokeExpr) {
                    SootMethod invokeMethod = ((InvokeExpr) rightOp).getMethod();
                    Type invokeRet = invokeMethod.getReturnType();
                    if(checkReturnType(invokeRet) && !invokeMethod.getDeclaringClass().getName().equals("retrofit2.Retrofit")){
                        if(!invokeMethod.getSignature().contains("retrofit2.Retrofit$Builder: retrofit2.Retrofit build()")) {
                            List<RetrofitBuildPoint> points = findRetrofitBuildMethod(invokeMethod);
                            if (!points.isEmpty()) {
                                for (RetrofitBuildPoint pt : points) {
                                    RetrofitBuildPoint newRetrofitBuildPoint = new RetrofitBuildPoint(pt);
                                    if(!pt.urlFromParam)
                                        newRetrofitBuildPoint.baseUrl = pt.baseUrl;
                                    newRetrofitBuildPoint.setCurrentMethod(sootMethod);
                                    newRetrofitBuildPoint.setCreateUnit(unit);
                                    addValue(localToPoint, leftOp, newRetrofitBuildPoint);
                                    tryToSolveParam(newRetrofitBuildPoint, pt, invokeMethod.getSignature(), (InvokeExpr) rightOp);
                                    if(newRetrofitBuildPoint.getCreateClass() != null)
                                        addValue(RetrofitClassToBuildPoint, newRetrofitBuildPoint.getCreateClass(), newRetrofitBuildPoint);
                                    tryToAddResult(sootMethod, newRetrofitBuildPoint);
                                    addValue(localToPoint, leftOp, newRetrofitBuildPoint);
                                    result.add(newRetrofitBuildPoint);
                                    LOGGER.info("86: Method: {} , From Method: {}, new Point: {}", methodSig, invokeMethod.getSignature(), newRetrofitBuildPoint.toString());
                                }
                            }
                        }
                        //TODO return okhttp client method.
                    }
                    if(HttpClientFind.findResult.containsKey(invokeMethod.getSignature())) {
                        localToClient.put(leftOp, HttpClientFind.findResult.get(invokeMethod.getSignature()));
                    }
                    if(allFactoryClass.containsKey(invokeRet.toString())){
                        AbstractFactory factory = allFactoryClass.get(invokeRet.toString());
                        localToFactory.put(leftOp, factory);
                        continue;
                    }


                    if (rightOp instanceof InstanceInvokeExpr) {
                        Value base = ((InstanceInvokeExpr) rightOp).getBase();
                        InvokeExpr invokeExpr = (InstanceInvokeExpr) rightOp;
                        String sig = ((InstanceInvokeExpr) rightOp).getMethod().getSignature();
                        if (localToPoint.containsKey(base)) {
                            List<RetrofitBuildPoint> localPoints = localToPoint.get(base);
                            if (sig.contains("retrofit2.Retrofit: java.lang.Object create(java.lang.Class)")) {
                                Value arg = invokeExpr.getArg(0);
                                if (arg instanceof Constant) {
                                    String clsName = (String) SimulateUtil.getConstant(arg);
                                    for(RetrofitBuildPoint lp: localPoints){
                                        if(lp.getCreateClass()==null) {
                                            lp.setCreateClass(clsName);
                                            lp.classFromParam = false;
                                            addValue(RetrofitClassToBuildPoint, clsName, lp);
                                        }
                                    }
                                }
                                else if(arg instanceof Local){
                                    //TODO local
                                    boolean fromParam = false;
                                    HashMap<String, List<Integer>> intereInvoke = new HashMap<>();
                                    intereInvoke.put(sig, new ArrayList<>(List.of(0)));
                                    LOGGER.warn("[DO LOCAL TO FIND CLASS]: {} -> {}", sootMethod, invokeExpr);
                                    MethodLocal methodLocal = new MethodLocal(sootMethod, intereInvoke, 2);
                                    methodLocal.setGetResult(true);
                                    methodLocal.doAnalysis();
                                    if(methodLocal.getLocalFromParams().containsKey(invokeExpr.getArg(0))) {
                                        fromParam = true;
                                        for (RetrofitBuildPoint lp : localPoints) {
                                            lp.classFromParam = true;
                                            lp.classParam = methodLocal.getLocalFromParams().get(invokeExpr.getArg(0));
                                        }
                                    }
                                    currentValues.putAll(methodLocal.getInterestingParamString());

                                    if (currentValues.containsKey(sig) && !fromParam){
                                        if(currentValues.get(sig).containsKey(0)) {
                                            List<String> clsNames = currentValues.get(sig).get(0);
                                            for (String clsName : clsNames) {
                                                for (RetrofitBuildPoint lp : localPoints) {
                                                    lp.setCreateClass(clsName);
                                                    addValue(RetrofitClassToBuildPoint, clsName, lp);
                                                }
                                            }
                                        }
                                    }
                                }
                                if(leftOp instanceof Local){
                                    localToPoint.put(leftOp, localToPoint.remove(base));
                                }
                                else if(leftOp instanceof FieldRef){
                                    FieldToBuildPoint.put(((FieldRef) leftOp).getField(), localToPoint.get(base));
                                }

                            } else if (sig.contains("retrofit2.Retrofit$Builder: retrofit2.Retrofit$Builder baseUrl(java.lang.String)") || sig.contains("setBaseUrl(java.lang.String)")) {
                                Value arg = invokeExpr.getArg(0);
                                if (arg instanceof Constant) {
                                    String baseUrl = (String) SimulateUtil.getConstant(arg);
                                    for(RetrofitBuildPoint lp: localPoints){
                                        lp.setBaseUrl(baseUrl);
                                    }
                                }
                                else if(arg instanceof Local){
                                    boolean fromParam = false;
                                    HashMap<String, List<Integer>> intereInvoke = new HashMap<>();
                                    intereInvoke.put(sig, new ArrayList<>(List.of(0)));
                                    MethodLocal methodLocal = new MethodLocal(sootMethod, intereInvoke, 2);
                                    methodLocal.setGetResult(true);
                                    methodLocal.doAnalysis();
                                    if(methodLocal.getLocalFromParams().containsKey(invokeExpr.getArg(0))) {
                                        fromParam = true;
                                        for (RetrofitBuildPoint lp : localPoints) {
                                            lp.urlFromParam = true;
                                            lp.urlParam = methodLocal.getLocalFromParams().get(invokeExpr.getArg(0));
                                            lp.baseUrl = new ArrayList<>();
                                        }
                                    }
                                    currentValues.putAll(methodLocal.getInterestingParamString());
                                    LOGGER.info("166: Do MethodLocal result: {}", currentValues.get(sig));

                                    if(currentValues.containsKey(sig)) {
                                        if (ret.toString().equals("void")) {
                                            if (currentValues.get(sig).containsKey(0)) {
                                                List<String> baseUrls = currentValues.get(sig).get(0);
                                                for (String baseUrl : baseUrls) {
                                                    for (RetrofitBuildPoint lp : localPoints) {
                                                        lp.setBaseUrl(baseUrl);
                                                        lp.urlFromParam = false;
                                                        lp.urlParam = new ArrayList<>();
                                                    }
                                                }
                                            }
                                        } else if (!fromParam) {
                                            if (currentValues.get(sig).containsKey(0)) {
                                                List<String> baseUrls = currentValues.get(sig).get(0);
                                                LOGGER.info("169: Do MethodLocal result: {}", currentValues.get(sig));
                                                for (String baseUrl : baseUrls) {
                                                    for (RetrofitBuildPoint lp : localPoints) {
                                                        lp.setBaseUrl(baseUrl);
                                                    }
                                                }
                                            }
                                        }
                                    }
                                }
                            } else if(sig.contains("Builder: retrofit2.Retrofit build()")){
                                localToPoint.put(leftOp, localToPoint.remove(base));

                            } else if (sig.contains("retrofit2.Retrofit$Builder: retrofit2.Retrofit$Builder client") || sig.contains("setOkHttpClient(okhttp3.OkHttpClient)")) {
                                //TODO client
                                Value arg = invokeExpr.getArg(0);
                                if (arg instanceof Local) {
                                    if(localToClient.containsKey(arg)){
                                        for (RetrofitBuildPoint lp : localPoints) {
                                            lp.setOkHttpClients(localToClient.get(arg));
                                        }
                                        LOGGER.info("GET LOCAL CLIENT: {}->{}",arg, localToClient.get(arg));
                                    } else if (HttpClientFind.findResult.containsKey(methodSig)) {
                                        List<AbstractHttpClient> tempClients = new ArrayList<>();
                                        for(AbstractHttpClient client: HttpClientFind.findResult.get(methodSig)){
                                            okHttpClient okClient = (okHttpClient) client;
                                            if (okClient.getLocalValue().equals(arg)){
                                                tempClients.add(okClient);
                                            }
                                            LOGGER.info("GET LOCAL CLIENT FROM CURRENT METHOD: {}->{}: {}", okClient.getLocalValue(), arg, okClient.toString());
                                        }
                                        if(!tempClients.isEmpty()){
                                            for (RetrofitBuildPoint lp : localPoints) {
                                                lp.setOkHttpClients(tempClients);
                                            }
                                        }
                                        else{
                                            LOGGER.info("NO LOCAL CLIENT: {}-> {}", arg, invokeExpr);
                                        }
                                    }
                                    // if local from okhttp build, then get result of this method
                                    // else if local from (return okhttp client) method, then get result of the return method.
                                    
                                }
                            } else if (sig.contains("retrofit2.Retrofit$Builder: retrofit2.Retrofit$Builder") && (sig.contains("ConverterFactory") || sig.contains("callFactory"))) {
                                Value arg0 = invokeExpr.getArg(0);
                                if(localToFactory.containsKey(arg0)){
                                    AbstractFactory abstractFactory = localToFactory.get(arg0);
                                    if(abstractFactory instanceof ConverterFactory){
                                        for(RetrofitBuildPoint localPoint : localPoints)
                                            localPoint.addConverterFactory(abstractFactory);
                                    }
                                    else{
                                        for(RetrofitBuildPoint localPoint : localPoints)
                                            localPoint.addCallFactory(abstractFactory);
                                    }
                                }
                                //TODO converterFactory

                            } else if(invokeExpr instanceof InterfaceInvokeExpr) {
                                if(RetrofitClassesWithMethods.containsKey(invokeMethod.getDeclaringClass())){
                                    RetrofitPoint retrofitPoint = getRPByMethod(invokeMethod);
                                    if(retrofitPoint != null){
                                        addValue(retrofitPoint.callByMethod, sootMethod);
                                        retrofitPoint.retrofitBuildPoint = localToPoint.get(base).get(0);
                                    }
                                }
                                else if(sig.contains("kotlin.Lazy: java.lang.Object getValue")){
                                    localToPoint.put(leftOp, localPoints);
                                }
                            }
                        }
                        else if (localToClient.containsKey(base)) {
                            localToClient.put(leftOp, localToClient.get(base));
                        }
                    }
                    else if(rightOp instanceof StaticInvokeExpr){
                        if(invokeMethod.getSignature().contains("kotlin.Lazy lazy(kotlin.jvm.functions.Function0)")){
                            Value para = ((StaticInvokeExpr) rightOp).getArg(0);
                            if(localToPoint.containsKey(para)){
                                localToPoint.put(leftOp, localToPoint.get(para));
                            }
                        }
                    }
                } else if (rightOp instanceof CastExpr) {
                    Value op = ((CastExpr) rightOp).getOp();
                    if(localToPoint.containsKey(op)){
                        localToPoint.put(leftOp, localToPoint.get(op));
                    }
                    if(localToClient.containsKey(op)){
                        localToClient.put(leftOp, localToClient.get(op));
                    }
                } else if(rightOp instanceof Local && localToPoint.containsKey(rightOp)){
                    RetrofitBuildPoint point = localToPoint.get(rightOp).get(0);
                    if(leftOp instanceof FieldRef){
                        SootField field = ((FieldRef) leftOp).getField();
                        if(FieldToBuildPoint.containsKey(field)){
                            for(RetrofitBuildPoint retrofitBuildPoint : FieldToBuildPoint.get(field)){
                                if(retrofitBuildPoint.getBaseUrl().isEmpty() && !point.getBaseUrl().isEmpty()) {
                                    retrofitBuildPoint.baseUrl = point.getBaseUrl();
                                }
                                if(!retrofitBuildPoint.classFromParam && retrofitBuildPoint.classParam == null && point.classParam !=null)
                                    retrofitBuildPoint.classParam = point.classParam;
                                if(retrofitBuildPoint.getOkHttpClients().isEmpty() && !point.getOkHttpClients().isEmpty()) {
                                    for(okHttpClient okClient : point.getOkHttpClients())
                                        retrofitBuildPoint.setOkHttpClient(okClient);
                                }
                                if(!point.getCallFactory().isEmpty()){
                                    addValue(retrofitBuildPoint.getCallFactory(), point.getCallFactory());
                                }
                                if(!point.getConverterFactory().isEmpty()){
                                    addValue(retrofitBuildPoint.getConverterFactory(), point.getConverterFactory());
                                }
                            }
                        }
                        else {
                            FieldToBuildPoint.put(field, localToPoint.get(rightOp));
                        }
                    }
                } else if(rightOp instanceof FieldRef){
                    SootField field = ((FieldRef) rightOp).getField();
                    if(FieldToBuildPoint.containsKey(field)) {
                        if (leftOp instanceof Local) {
                            localToPoint.put(leftOp, FieldToBuildPoint.get(field));
                            addValue(result, FieldToBuildPoint.get(field));
                        }
                    }
                    else{
                        if((field.getType() instanceof RefType)){
                            RefType refType = (RefType) field.getType();
                            SootClass refClass = refType.getSootClass();
                            if(RetrofitClassesWithMethods.containsKey(refClass) || refType.getClassName().contains("retrofit2.Retrofit") || refType.getClassName().contains("kotlin.Lazy")){
                                SootClass sootClass = sootMethod.getDeclaringClass();
                                for (SootMethod method : clone(sootClass.getMethods())) {
                                    if(method.equals(sootMethod)) continue;
                                    if (!method.isConcrete()) continue;
                                    findRetrofitBuildMethod(method);
                                }
                                if(FieldToBuildPoint.containsKey(field)) {
                                    if (leftOp instanceof Local) {
                                        localToPoint.put(leftOp, FieldToBuildPoint.get(field));
                                        addValue(result, FieldToBuildPoint.get(field));
                                    }
                                }
                                else {
                                    RetrofitBuildPoint retrofitBuildPoint = new RetrofitBuildPoint(sootMethod, unit);
                                    addValue(FieldToBuildPoint, field, retrofitBuildPoint);
                                    addValue(localToPoint, leftOp, retrofitBuildPoint);
                                    result.add(retrofitBuildPoint);
                                }
                            }
                            else{
                                if(checkIfKotlinFunction0(refClass)){
                                    if(refClass.getName().contains("com.liesheng.module_device.setting.viewmodel.OtaCheckViewModel$deviceApi$2"))
                                        LOGGER.info("got");
                                    SootMethod invokeM = refClass.getMethod("java.lang.Object invoke()");
                                    if(invokeM != null){
                                        findRetrofitBuildMethod(invokeM);
                                        if(findResult.containsKey(invokeM.getSignature())){
                                            if (leftOp instanceof Local) {
                                                localToPoint.put(leftOp, findResult.get(invokeM.getSignature()));
                                                addValue(result, findResult.get(invokeM.getSignature()));
                                            }
                                        }
                                    }
                                }
                            }
                        }
                    }
                    if(HttpClientFind.FieldToClientPoint.containsKey(field)){
                        if(leftOp instanceof Local) {
                            localToClient.put(leftOp, HttpClientFind.FieldToClientPoint.get(((FieldRef) rightOp).getField()));
                        }
                    }
                }
            }
            else if(stmt instanceof InvokeStmt){
                if(stmt.getInvokeExpr() instanceof InstanceInvokeExpr) {
                    Value base = ((InstanceInvokeExpr) stmt.getInvokeExpr()).getBase();
                    InvokeExpr invokeExpr =  stmt.getInvokeExpr();
                    String sig = (stmt.getInvokeExpr()).getMethod().getSignature();
                    if (localToPoint.containsKey(base)) {
                        List<RetrofitBuildPoint> localPoints = localToPoint.get(base);
                        if (sig.contains("retrofit2.Retrofit$Builder: retrofit2.Retrofit$Builder client")) {
                            //TODO client
                            Value arg = invokeExpr.getArg(0);
                            if (arg instanceof Local) {
                                if (localToClient.containsKey(arg)) {
                                    for (RetrofitBuildPoint lp : localPoints) {
                                        lp.setOkHttpClients(localToClient.get(arg));
                                    }

                                } else if (HttpClientFind.findResult.containsKey(methodSig)) {
                                    List<AbstractHttpClient> tempClients = new ArrayList<>();
                                    for (AbstractHttpClient client : HttpClientFind.findResult.get(methodSig)) {
                                        okHttpClient okClient = (okHttpClient) client;
                                        if (okClient.getLocalValue().equals(arg)) {
                                            tempClients.add(okClient);
                                        }
                                    }
                                    if (!tempClients.isEmpty()) {
                                        for (RetrofitBuildPoint lp : localPoints) {
                                            lp.setOkHttpClients(tempClients);
                                        }
                                    }
                                }
                                // if local from okhttp build, then get result of this method
                                // else if local from (return okhttp client) method, then get result of the return method.

                            }
                        }
                    }

                }
            }
            else if(stmt instanceof ReturnStmt){
                Value op = ((ReturnStmt) stmt).getOp();
                if(localToPoint.containsKey(op)){
                    addValue(result, localToPoint.get(op));
                }
            }
        }
        if(!result.isEmpty())
            tryToAddResult(sootMethod, result);

        return result;
    }

    private static void tryToSolveParam(RetrofitBuildPoint newPoint, RetrofitBuildPoint oldPoint, String sig, InvokeExpr invokeExpr){
        HashMap<String, List<Integer>> interestInvoke = new HashMap<>();
        interestInvoke.put(sig, new ArrayList<>());
        if(oldPoint.urlFromParam){
            addValue(interestInvoke, sig, oldPoint.urlParam);
        }
        if(oldPoint.classFromParam){
            addValue(interestInvoke, sig, oldPoint.classParam);
        }
        if(!interestInvoke.get(sig).isEmpty()) {
            MethodLocal methodLocal = new MethodLocal(newPoint.getCurrentMethod(), interestInvoke, 2);
            methodLocal.setGetResult(true);
            methodLocal.doAnalysis();
            HashMap<Value, List<Integer>> localFromParams = methodLocal.getLocalFromParams();
            HashMap<String, HashMap<Integer, List<String>>> result = methodLocal.getInterestingParamString();
            if(result.containsKey(sig)) {
                for(int i : result.get(sig).keySet()) {
                    if (oldPoint.urlFromParam && oldPoint.urlParam.contains(i)) {
                        if (localFromParams.containsKey(invokeExpr.getArg(i))) {
                            newPoint.urlFromParam = true;
                            newPoint.urlParam = localFromParams.get(invokeExpr.getArg(i));
                        } else if (result.get(sig).containsKey(i)) {
                            List<String> urlResult = result.get(sig).get(i);
                            for (String url : urlResult)
                                newPoint.setBaseUrl(url);
                        }
                    }
                    if (oldPoint.classFromParam && oldPoint.classParam.contains(i)) {
                        if (localFromParams.containsKey(invokeExpr.getArg(i))) {
                            newPoint.classFromParam = true;
                            newPoint.classParam = localFromParams.get(invokeExpr.getArg(i));
                        } else if (result.get(sig).containsKey(i)) {
                            List<String> classResult = result.get(sig).get(i);
                            if (!classResult.isEmpty()) {
                                newPoint.setCreateClass(classResult.get(0));
                            }
                        }
                    }
                }
            }

        }
    }

    private static void tryToAddResult(SootMethod sootMethod, RetrofitBuildPoint point){
        addValue(findResult, sootMethod.getSignature(), point);
    }

    private static void tryToAddResult(SootMethod sootMethod, List<RetrofitBuildPoint> points){
        for (RetrofitBuildPoint point : points)
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

//    public static HashMap<Integer, String> checkArgType(InvokeExpr invokeExpr){
//        List<Value> args = invokeExpr.getArgs();
//        HashMap<Integer, String> result = new HashMap<>();
//        String clsString = "";
//        int i = 0;
//        for(Value arg : args){
//            if(arg instanceof ClassConstant){
//                clsString = (String)SimulateUtil.getConstant(arg);
//                result.put(i,clsString);
//            }
//            i++;
//        }
//        return result;
//    }

    public static boolean checkReturnType(Type v){
        String typeClz = v.toString();
        for(SootClass clz : RetrofitClassesWithMethods.keySet()){
            if(typeClz.equals(clz.getName()))
                return true;
        }
        return typeClz.equals("void") ||typeClz.equals("java.lang.Object") || typeClz.contains("retrofit2.Retrofit");
    }

    public static boolean checkIfKotlinFunction0(SootClass clz){
        boolean result = false;
        for(SootClass interfaceClz : clz.getInterfaces()){
            if(interfaceClz.getName().equals("kotlin.jvm.functions.Function0"))
                result = true;
        }
        return result;
    }


    public static RetrofitPoint getRPByMethod(SootMethod method){
        SootClass sootClass = method.getDeclaringClass();
        List<RetrofitPoint> retrofitPoints = RetrofitClassesWithMethods.get(sootClass);
        if(retrofitPoints != null){
            for(RetrofitPoint retrofitPoint: retrofitPoints){
                if(retrofitPoint.getMethod().equals(method))
                    return retrofitPoint;
            }
        }
        return null;
    }

    public static <T> List<T> clone(List<T> ls) {
        return new ArrayList<T>(ls);
    }


}
