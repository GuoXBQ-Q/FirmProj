package firmproj.objectSim;

import firmproj.base.MethodString;
import firmproj.base.ValueContext;
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

public class StringClz implements AbstractClz{
    private final SootClass currentClass;
    private final SootMethod ParentMethod;
    private List<String> result = new ArrayList<>();
    private final List<ValueContext> valueContexts = new ArrayList<>();
    private boolean solved = false;

    public StringClz(SootClass currentClass, SootMethod method){
        this.currentClass = currentClass;
        this.ParentMethod = method;
    }

    public StringClz(SootClass currentClass, SootMethod method, List<ValueContext> values){
        this(currentClass, method);
        this.valueContexts.addAll(values);
    }

    @Override
    public void solve() {
        List<String> tmpResult = new ArrayList<>();
        for(ValueContext vc : valueContexts){
            Unit u = vc.getCurrentUnit();
            if(u instanceof AssignStmt){
                Value rightOp = ((AssignStmt) u).getRightOp();
                if(rightOp instanceof InvokeExpr){
                    InvokeExpr invokeExpr = (InvokeExpr) rightOp;
                    SootMethod method = invokeExpr.getMethod();
                    String sig = method.getSignature();
                    if(sig.contains("java.lang.StringBuilder: java.lang.StringBuilder append")||
                            sig.contains("java.lang.StringBuffer: java.lang.StringBuffer append")||
                            sig.contains("java.lang.String: java.lang.String concat")){
                        HashMap<Value, List<String>> currentValues = vc.getCurrentValues();
                        int argIndex = 0;
                        List<List<String>> args = new ArrayList<>();
                        for(Value value: invokeExpr.getArgs()){
                            if(value instanceof Constant) {
                                Object constObj = SimulateUtil.getConstant(value);
                                if(constObj != null) {
                                    if (constObj.toString().equals("58"))
                                        args.add(List.of(":"));
                                    else {
                                        args.add(List.of(constObj.toString()));
                                    }
                                }
                                else{
                                    args.add(List.of("null"));
                                }
                            }
                            else{
                                if(currentValues.get(value).toString().length() > 2000) continue;
                                args.add(currentValues.get(value));
                            }
                            argIndex++;
                        }
                        if(args.get(0) != null)
                            tmpResult.add(MethodString.getContent(args.get(0)));
                    }
                    else if(sig.contains("java.lang.String: java.lang.String toLowerCase")){
                        tmpResult.replaceAll(String::toLowerCase);
                    } else if (sig.contains("java.lang.String: java.lang.String toUpperCase")) {
                        tmpResult.replaceAll(String::toUpperCase);
                    } else if(sig.contains("java.lang.String: java.lang.String replace")){
                        tmpResult.replaceAll(s -> s.replace(MethodString.getContent(invokeExpr.getArg(0)), MethodString.getContent(invokeExpr.getArg(1))));
                    }
                }
            }
            else if (u instanceof InvokeStmt){
                InvokeExpr invokeExpr = ((InvokeStmt) u).getInvokeExpr();
                SootMethod method = ((InvokeStmt) u).getInvokeExpr().getMethod();
                String sig = method.getSignature();
                if(sig.contains("java.lang.StringBuilder: java.lang.StringBuilder append")||
                        sig.contains("java.lang.StringBuffer: java.lang.StringBuffer append")||
                        sig.contains("java.lang.String: java.lang.String concat")){
                    HashMap<Value, List<String>> currentValues = vc.getCurrentValues();
                    int argIndex = 0;
                    List<List<String>> args = new ArrayList<>();
                    for(Value value: invokeExpr.getArgs()){
                        if(value instanceof Constant) {
                            Object constObj = SimulateUtil.getConstant(value);
                            if(constObj != null) {
                                if (constObj.toString().equals("58"))
                                    args.add(List.of(":"));
                                args.add(List.of(constObj.toString()));
                            }
                            else{
                                args.add(List.of("null"));
                            }
                        }
                        else{
                            if(currentValues.get(value).toString().length() > 2000) continue;
                            args.add(currentValues.get(value));
                        }
                        argIndex++;
                    }
                    if(args.get(0) != null)
                        tmpResult.add(MethodString.getContent(args.get(0)));
                }
            }
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
    public SootClass getCurrentClass(){
        return this.currentClass;
    }

    @Override
    public List<ValueContext> getCurrentValues() {
        return this.valueContexts;
    }

    @Override
    public List<String> getResult() {
        return result;
    }

    @Override
    public boolean isSolved() {
        return solved;
    }

    @Override
    public String toString() {
        return "StringClz{" +
                "result=" + MethodString.getContent(result) +
                '}';
    }
}
