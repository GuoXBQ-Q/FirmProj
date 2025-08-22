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

import java.util.*;

public class JSONObjectClz implements AbstractClz{

    private final SootClass currentClass;
    private final SootMethod ParentMethod;
    private HashMap<List<String>, List<String>> result = new LinkedHashMap<>();
    private final List<ValueContext> valueContexts = new ArrayList<>();
    private boolean solved = false;

    public JSONObjectClz(SootClass currentClass, SootMethod method){
        this.currentClass = currentClass;
        this.ParentMethod = method;
    }

    public JSONObjectClz(SootClass currentClass, SootMethod method, List<ValueContext> values){
        this(currentClass, method);
        this.valueContexts.addAll(values);
    }

    @Override
    public void solve() {
        HashMap<List<String>,List<String>> tmpResult = new LinkedHashMap<>();
        for(ValueContext vc : valueContexts){
            Unit u = vc.getCurrentUnit();
            if(u instanceof AssignStmt){
                Value rightOp = ((AssignStmt) u).getRightOp();
                if(rightOp instanceof InvokeExpr){
                    InvokeExpr invokeExpr = (InvokeExpr) rightOp;
                    SootMethod method = invokeExpr.getMethod();
                    if(method.getSignature().contains("JSONObject put(java.lang.String,java.lang.Object)")){
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
                        tmpResult.put(args.get(0), args.get(1));
                    }
                }
            }
            else if (u instanceof InvokeStmt){
                InvokeExpr invokeExpr = ((InvokeStmt) u).getInvokeExpr();
                SootMethod method = ((InvokeStmt) u).getInvokeExpr().getMethod();
                if(method.getSignature().contains("JSONObject put(java.lang.String,java.lang.Object)")){
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
                    tmpResult.put(args.get(0), args.get(1));
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

    public boolean isSolved() {
        return solved;
    }

    @Override
    public HashMap<List<String>, List<String>> getResult() {
        return result;
    }

    @Override
    public String toString() {
        solve();
        StringBuilder result = new StringBuilder();
        result.append("MapCLZ: From Method: ").append(this.ParentMethod.getSubSignature()).append(" ");
        result.append("Map Entry: {");
        for(Map.Entry<List<String>, List<String>> entry: this.result.entrySet()){
            result.append(MethodString.getContent(entry.getKey()));
            result.append('=');
            result.append(MethodString.getContent(entry.getValue()));
            result.append(",");
        }
        result.deleteCharAt(result.length()-1);
        if(!this.result.isEmpty())
            result.append("} ");
        return result.toString();
    }
}
