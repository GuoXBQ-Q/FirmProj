package firmproj.graph;

import firmproj.base.MethodString;
import org.apache.logging.log4j.LogManager;
import org.apache.logging.log4j.Logger;
import soot.*;
import soot.jimple.*;
import soot.tagkit.InnerClassTag;
import soot.tagkit.Tag;
import soot.toolkits.graph.Block;
import soot.toolkits.graph.CompleteBlockGraph;
import soot.util.Chain;
import java.util.*;

/**
 * Creates the CallGraph of the application under analysis
 */
public class CallGraph {

    private static final Logger LOGGER = LogManager.getLogger(CallGraph.class);

    //Key the string of the soot method
    // map with the sootMethod name and the CallGraphNode of the sootMethod
    private static final Hashtable<String, CallGraphNode> nodes = new Hashtable<>();

    public static final HashMap<String, List<SootClass>> outerInnerClasses = new HashMap<>();

    public static final HashMap<SootClass, CallBackPoint> callBackClassesWithPoint = new HashMap<>();
    // Maps Soot Field String to the soot Methods it is referenced

    public static void init() {
        long startTime = System.currentTimeMillis();

        Chain<SootClass> classes = Scene.v().getClasses();
        try {
            //init the nodes map
            for (SootClass sootClass : classes) {
                if(MethodString.isStandardLibraryClass(sootClass)) continue;
                if(sootClass.hasTag("InnerClassTag")){
                    Tag tag = sootClass.getTag("InnerClassTag");
                    if(tag instanceof InnerClassTag){
                        if(((InnerClassTag) tag).getOuterClass() != null){
                            String outerClass = ((InnerClassTag) tag).getOuterClass().replace('/','.');
                            if(!outerInnerClasses.containsKey(outerClass))
                                outerInnerClasses.put(outerClass, new ArrayList<>());
                            if(!outerInnerClasses.get(outerClass).contains(sootClass)) {
                                outerInnerClasses.get(outerClass).add(sootClass);
                                if(!MethodString.classWithOuterInnerFields.containsKey(outerClass))
                                    MethodString.classWithOuterInnerFields.put(outerClass, new ArrayList<>());
                                List<SootField> fields = MethodString.classWithOuterInnerFields.get(outerClass);
                                for(SootField field: sootClass.getFields()) {
                                    if(!fields.contains(field))
                                        fields.add(field);
                                }
                            }
                            for(SootClass clz : sootClass.getInterfaces()){
                                String clzName = clz.getName();
                                if(clzName.toLowerCase().contains("callback")){
                                    CallBackPoint callBackPoint = new CallBackPoint(sootClass);
                                    callBackClassesWithPoint.put(sootClass, callBackPoint);
                                }
                            }
                            String clzName = sootClass.getSuperclass().getName();
                            if(clzName.toLowerCase().contains("callback")){
                                CallBackPoint callBackPoint = new CallBackPoint(sootClass);
                                callBackClassesWithPoint.put(sootClass, callBackPoint);
                            }
                        }
                    }
                }
                List<SootMethod> methods = new ArrayList<>(sootClass.getMethods());
                for (SootMethod sootMethod : methods) {
                    CallGraphNode tmpNode = new CallGraphNode(sootMethod);
                    nodes.put(sootMethod.toString(), tmpNode);
                    if (sootMethod.isConcrete()) {
                        try {
                            sootMethod.retrieveActiveBody();
                        } catch (Exception e) {
                            LOGGER.error("Could not retrieved the active body of {} because {}", sootMethod, e.getLocalizedMessage());
                        }
                    }
                }
            }

            LOGGER.debug("[CG time]: " + (System.currentTimeMillis() - startTime));
            for (SootClass sootClass : classes) {
                //LOGGER.warn(sootClass.getName());
                boolean needCheck = !MethodString.isStandardLibraryClass(sootClass);
                if(!needCheck) continue;
                for (SootMethod sootMethod : clone(sootClass.getMethods())) {
                    if (!sootMethod.isConcrete())
                        continue;
                    Body body = null;
                    try {
                        body = sootMethod.getActiveBody();
                    } catch (Exception e) {
                        LOGGER.error("Could not retrieved the active body {} because {}", sootMethod, e.getLocalizedMessage());
                    }
                    if (body == null)
                        continue;

                   HashMap<Value, SootMethod> localToInterface = new HashMap<>();
                   HashMap<Value, CallBackPoint> localToCallBack = new HashMap<>();
                    for (Unit unit : body.getUnits()) {
                        if (unit instanceof Stmt) {
                            if (((Stmt) unit).containsInvokeExpr()) {
                                try {
                                    addCall(sootMethod, ((Stmt) unit).getInvokeExpr().getMethod());
                                } catch (Exception e) {
                                    LOGGER.error(e.getMessage());
                                }
                            }
                            if(unit instanceof AssignStmt ){
                                Value leftOp = ((AssignStmt) unit).getLeftOp();
                                Value rightOp = ((AssignStmt) unit).getRightOp();
                                if(rightOp instanceof NewExpr) {
                                    NewExpr newExpr = (NewExpr) ((AssignStmt) unit).getRightOp();
                                    SootClass cls = newExpr.getBaseType().getSootClass();
                                    if (callBackClassesWithPoint.containsKey(cls)) {
                                        CallBackPoint callBackPoint = callBackClassesWithPoint.get(cls);
                                        localToCallBack.put(leftOp, callBackPoint);
                                        if (!callBackPoint.parentMethod.contains(sootMethod))
                                            callBackPoint.parentMethod.add(sootMethod);
                                    }
                                }
                                else if(rightOp instanceof InterfaceInvokeExpr){
                                    if(((InterfaceInvokeExpr) rightOp).getMethod().getReturnType().toString().contains("retrofit2.Call")){
                                        SootMethod interfaceMethod = ((InterfaceInvokeExpr) rightOp).getMethod();
                                        localToInterface.put(((AssignStmt) unit).getLeftOp(), interfaceMethod);
                                    }
                                }
                            }
                            else if(unit instanceof InvokeStmt){
                                InvokeExpr invokeExpr = ((InvokeStmt) unit).getInvokeExpr();
                                if(invokeExpr instanceof InterfaceInvokeExpr) {
                                    Value base = ((InterfaceInvokeExpr) invokeExpr).getBase();
                                    if (invokeExpr.getMethod().getSignature().contains("retrofit2.Call: void enqueue")) {
                                        if (localToInterface.containsKey(base)) {
                                            Value arg = invokeExpr.getArg(0);
                                            if(localToCallBack.containsKey(arg)) {
                                                MethodString.addValue(localToCallBack.get(arg).relatedPoint, localToInterface.get(base));
                                            }
                                        }
                                    }
                                }
                            }
                        }

                    }
                }
            }
        } catch (Throwable e) {
            LOGGER.error( e.getLocalizedMessage()+"error init call graph");
        }


        LOGGER.debug("[CG time3]:" + (System.currentTimeMillis() - startTime));
        LOGGER.info("CallbackCLass: {}", callBackClassesWithPoint);
        //LOGGER.info("total: " + visitedMethod.toString());
    }

    /**
     * Add to the call graph nodes the information about the callee and caller
     *
     * @param from add to the from node the call information
     * @param to   add to the to node the caller information
     */
    private static void addCall(SootMethod from, SootMethod to) {
        CallGraphNode fromNode, toNode;
        fromNode = getNode(from);
        toNode = getNode(to);
        if (fromNode == null || toNode == null) {
            //LOGGER.debug("Can't add call because from or to node is null");
            return;
        }

        fromNode.addCallTo(toNode);
        toNode.addCallBy(fromNode);

    }

    /**
     * get CallGraphNode from Soot Method
     *
     * @param from to get CallGraphNode from
     * @return the corresponding node
     */
    public static CallGraphNode getNode(SootMethod from) {
        return getNode(from.toString());
    }

    /**
     * get CallGraphNode from Soot Method
     *
     * @param from SootMethodString to get the CallGraphNode from
     * @return the corresponding node
     */
    public static CallGraphNode getNode(String from) {
        return nodes.get(from);
    }


    public static <T> List<T> clone(List<T> ls) {
        return new ArrayList<T>(ls);
    }

    /**
     * Search for the callers of the passed signature
     *
     * @param signature of the soot method
     * @return found callers
     */
    public static List<CallGraphNode> findCaller(String signature, boolean findSubMethods) {
        SootMethod sootMethod = null;
        try {
            sootMethod = Scene.v().getMethod(signature);

        } catch (RuntimeException e) {
            LOGGER.debug(String.format("%s not found", signature));
            sootMethod = findParentMethod(signature);
            findSubMethods = true;
        }

        if (sootMethod == null) {
            return new LinkedList<>();
        }

        if(MethodString.isStandardLibraryClass(sootMethod.getDeclaringClass())) return new ArrayList<>();

        HashSet<SootMethod> methods = new HashSet<>();
        methods.add(sootMethod);
        if (sootMethod.getName().charAt(0) != '<') {
            findAllPointerOfThisMethod(methods, sootMethod.getSubSignature(), sootMethod.getDeclaringClass());
        }
        if (findSubMethods) {
            findAllSubPointerOfThisMethod(methods, sootMethod.getSubSignature(), sootMethod.getDeclaringClass());
        }
        HashSet<CallGraphNode> callGraphNodes = new HashSet<>();
        for (SootMethod tmpMethod : methods) {
            CallGraphNode node = CallGraph.getNode(tmpMethod.toString());
            if (node == null) {
                LOGGER.debug("CallGraph node is null");
                continue;
            }
            callGraphNodes.addAll(node.getCallBy());

        }
        return new ArrayList<>(callGraphNodes);
    }

    /**
     * search for methods implementing signature in case it is an Interface/abstract class?
     *
     * @param methods   resulting methods
     * @param subSig    of the method searched
     * @param sootClass where to look for the method signature
     */
    private static void findAllPointerOfThisMethod(HashSet<SootMethod> methods, String subSig, SootClass sootClass) {
        if (sootClass == null) {
            LOGGER.debug("Soot Class is null in findAllPointerOfThisMethod");
            return;
        }
        if (MethodString.isStandardLibraryClass(sootClass)) {
            return;
        }
        try {
            if (sootClass.getMethod(subSig) != null) {
                methods.add(sootClass.getMethod(subSig));
            }
        } catch (Exception e) {
            //LOGGER.debug("Caught the exception {}", e.getLocalizedMessage());
        }

        if (sootClass.getSuperclass() != sootClass && sootClass.getSuperclass() != null) {
            //no multiple superclasses possible in java
            findAllPointerOfThisMethod(methods, subSig, sootClass.getSuperclass());
        }
        for (SootClass itf : sootClass.getInterfaces()) {
            findAllPointerOfThisMethod(methods, subSig, itf);
        }
    }

    public static void findAllSubPointerOfThisMethod(HashSet<SootMethod> methods, String subSig, SootClass sootClass) {
        HashSet<SootClass> sootClasses = new HashSet<>();
        try {
            sootClasses.addAll(Scene.v().getActiveHierarchy().getSubclassesOf(sootClass));
        } catch (Exception e) {
            LOGGER.debug("Could not get subclasses");
        }
        try {
            sootClasses.addAll(Scene.v().getActiveHierarchy().getSubinterfacesOf(sootClass));
        } catch (Exception e) {
            LOGGER.debug("Could not get subinterfaces");
        }

        if(sootClasses.isEmpty()) return;

        sootClasses.forEach(x -> {
            try {
                methods.add(x.getMethod(subSig));
            } catch (Exception e) {
                LOGGER.debug("Method not implemented");
            }
        });
    }

    private static SootMethod findParentMethod(String signature) {
        SootMethod result = null;

        try {
            String className = signature.substring(signature.indexOf("<")+1, signature.indexOf(":")).trim();
            String methodSubName = signature.substring(signature.indexOf(":")+1).replace(">", "").trim();
            SootClass sootClass = Scene.v().getSootClass(className);
            while (!sootClass.getName().equals("java.lang.Object")) {
                try {
                    result = sootClass.getMethod(methodSubName);
                } catch (RuntimeException e) {
                    //LOGGER.debug(String.format("%s not found in %s", methodSubName, sootClass.getName()));
                }
                sootClass = sootClass.getSuperclass();
            }

        } catch (Throwable e) {
            //LOGGER.debug(String.format("could not find parent method of %s ", signature));

        }
        return result;
    }
}
