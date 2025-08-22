package firmproj.objectSim;

import firmproj.base.MethodString;
import firmproj.base.ValueContext;
import firmproj.client.CustomHttpClient;
import soot.SootClass;
import soot.SootMethod;
import soot.Unit;
import soot.Value;
import soot.jimple.AssignStmt;
import soot.jimple.Constant;
import soot.jimple.InvokeExpr;
import soot.jimple.InvokeStmt;

import java.util.ArrayList;
import java.util.HashMap;
import java.util.List;

public class UrlClz implements AbstractClz {

    private final SootClass currentClass;
    private final SootMethod ParentMethod;

    private List<String> result = new ArrayList<>();
    private final List<ValueContext> valueContexts = new ArrayList<>();
    private boolean solved = false;

    private boolean isUrlClient = false;
    HashMap<String, List<String>> RequestProperty = new HashMap<>();
    private final CustomHttpClient clientResult = new CustomHttpClient();

    public UrlClz(SootClass currentClass, SootMethod method){
        this.currentClass = currentClass;
        this.ParentMethod = method;
    }

    @Override
    public void solve() {
        List<String> tmpResult = new ArrayList<>();
        RequestProperty.clear();
        for(ValueContext vc : valueContexts){
            Unit u = vc.getCurrentUnit();
            if(u instanceof AssignStmt){
                Value rightOp = ((AssignStmt) u).getRightOp();
                if(rightOp instanceof InvokeExpr){
                    InvokeExpr invokeExpr = (InvokeExpr) rightOp;
                    SootMethod method = invokeExpr.getMethod();
                    String sig = method.getSignature();
                    if(sig.contains("java.net.URL: void <init>(java.lang.String)")){
                        HashMap<Value, List<String>> currentValues = vc.getCurrentValues();
                        int argIndex = 0;
                        List<List<String>> args = new ArrayList<>();
                        for(Value value: invokeExpr.getArgs()){
                            if(value instanceof Constant) {
                                Object constObj = SimulateUtil.getConstant(value);
                                if(constObj != null)
                                    args.add(List.of(constObj.toString()));
                                else{
                                    args.add(List.of("null"));
                                }
                            }
                            else{
                                args.add(currentValues.get(value));
                            }
                            argIndex++;
                        }
                        if(args.get(0) != null) {
                            tmpResult.add(MethodString.getContent(args.get(0)));
                            RequestProperty.put("url", new ArrayList<>(args.get(0)));
                            if(args.get(0).contains("$[")){
                                clientResult.setNeedRequestContent(true);
                            }
                        }
                    }
                    else if(sig.contains("URLConnection openConnection()")){
                        isUrlClient = true;
                    }
                }
            }
            else if (u instanceof InvokeStmt){
                InvokeExpr invokeExpr = ((InvokeStmt) u).getInvokeExpr();
                SootMethod method = ((InvokeStmt) u).getInvokeExpr().getMethod();
                String sig = method.getSignature();
                if(sig.contains("java.net.URL: void <init>(java.lang.String)")){
                    HashMap<Value, List<String>> currentValues = vc.getCurrentValues();
                    int argIndex = 0;
                    List<List<String>> args = new ArrayList<>();
                    for(Value value: invokeExpr.getArgs()){
                        if(value instanceof Constant) {
                            Object constObj = SimulateUtil.getConstant(value);
                            if(constObj != null)
                                args.add(List.of(constObj.toString()));
                            else{
                                args.add(List.of("null"));
                            }
                        }
                        else{
                            args.add(currentValues.get(value));
                        }
                        argIndex++;
                    }
                    if(args.get(0) != null) {
                        tmpResult.add(MethodString.getContent(args.get(0)));
                        RequestProperty.put("url", new ArrayList<>(args.get(0)));
                        if(args.get(0).contains("$[")){
                            clientResult.setNeedRequestContent(true);
                        }
                    }
                }
                else if(sig.contains("URLConnection: void setRequestProperty(java.lang.String,java.lang.String)")){
                    HashMap<Value, List<String>> currentValues = vc.getCurrentValues();
                    int argIndex = 0;
                    List<List<String>> args = new ArrayList<>();
                    for(Value value: invokeExpr.getArgs()){
                        if(value instanceof Constant) {
                            Object constObj = SimulateUtil.getConstant(value);
                            if(constObj != null)
                                args.add(List.of(constObj.toString()));
                            else{
                                args.add(List.of("null"));
                            }
                        }
                        else{
                            args.add(currentValues.get(value));
                        }
                        argIndex++;
                    }
                    if(args.size() > 1){
                        try {
                            RequestProperty.put(args.get(0).get(0), args.get(1));
                        } catch (Throwable ignore){}
                    }

                }
                else if(sig.contains("URLConnection: void setRequestMethod")){
                    HashMap<Value, List<String>> currentValues = vc.getCurrentValues();
                    int argIndex = 0;
                    List<List<String>> args = new ArrayList<>();
                    for(Value value: invokeExpr.getArgs()){
                        if(value instanceof Constant) {
                            Object constObj = SimulateUtil.getConstant(value);
                            if(constObj != null)
                                args.add(List.of(constObj.toString()));
                            else{
                                args.add(List.of("null"));
                            }
                        }
                        else{
                            args.add(currentValues.get(value));
                        }
                        argIndex++;
                    }
                    if(args.size() == 1){
                        RequestProperty.put("Method", args.get(0));
                    }
                }
                else if(sig.contains("java.io.OutputStream: void write")){
                    HashMap<Value, List<String>> currentValues = vc.getCurrentValues();
                    int argIndex = 0;
                    List<List<String>> args = new ArrayList<>();
                    for(Value value: invokeExpr.getArgs()){
                        if(value instanceof Constant) {
                            Object constObj = SimulateUtil.getConstant(value);
                            if(constObj != null)
                                args.add(List.of(constObj.toString()));
                            else{
                                args.add(List.of("null"));
                            }
                        }
                        else{
                            args.add(currentValues.get(value));
                        }
                        argIndex++;
                    }
                    if(args.size() == 1){
                        RequestProperty.put("Body", args.get(0));
                    }
                }
            }
        }
        if(isUrlClient){
            clientResult.sootMethod = getParentMethod();
            if(!RequestProperty.isEmpty())
                clientResult.requestContentFromParams.putAll(RequestProperty);
        }
        result = tmpResult;
        solved = !result.isEmpty();
    }

    @Override
    public void init() {

    }

    @Override
    public void addValueContexts(ValueContext valueContext) {
        this.valueContexts.add(valueContext);
    }

    @Override
    public SootMethod getParentMethod() {
        return this.ParentMethod;
    }

    @Override
    public SootClass getCurrentClass() {
        return this.currentClass;
    }

    @Override
    public List<ValueContext> getCurrentValues() {
        return List.of();
    }

    @Override
    public Object getResult() {
        return result;
    }

    public CustomHttpClient getClientResult() {
        return clientResult;
    }

    @Override
    public boolean isSolved() {
        return solved;
    }

    public boolean isUrlClient() {
        return isUrlClient;
    }

    @Override
    public String toString() {
        return "UrlClz{" +
                "result=" + MethodString.getContent(result) +
                '}';
    }
}
