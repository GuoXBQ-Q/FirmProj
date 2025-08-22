package firmproj.base;

import soot.SootMethod;

import java.util.*;

public class MethodParamInvoke {
    public SootMethod sootMethod;
    public boolean solved;
    public List<Integer> param = new ArrayList<>();
    public HashMap<Integer, List<String>> paramValue = new HashMap<>();
    public final List<String> invokeMethodSig = new ArrayList<>();

    public MethodParamInvoke(SootMethod method, Integer para){
        sootMethod = method;
        param.add(para);
    }

    public MethodParamInvoke(SootMethod method, Integer para, String sig){
        sootMethod = method;
        param.add(para);
        invokeMethodSig.add(sig);
    }

    public MethodParamInvoke(SootMethod method, Integer para, List<String> sig){
        sootMethod = method;
        param.add(para);
        invokeMethodSig.addAll(sig);
    }

    public MethodParamInvoke(SootMethod method, List<Integer> para){
        sootMethod = method;
        param = para;
    }

    public MethodParamInvoke(SootMethod method, List<Integer> para, String sig){
        sootMethod = method;
        param = para;
        invokeMethodSig.add(sig);
    }

    public MethodParamInvoke(SootMethod method, List<Integer> para, List<String> sig){
        sootMethod = method;
        param = para;
        invokeMethodSig.addAll(sig);
    }

    public MethodParamInvoke(MethodParamInvoke OldmethodParamInvoke){
        this.sootMethod = OldmethodParamInvoke.sootMethod;
        this.param.addAll(OldmethodParamInvoke.param);
        this.paramValue.putAll(OldmethodParamInvoke.paramValue);
        this.solved = OldmethodParamInvoke.solved;
        this.invokeMethodSig.addAll(OldmethodParamInvoke.invokeMethodSig);
    }

    public void addMethodInvoke(String methodInvoke){
        for(String str : invokeMethodSig){
            if(str == null) continue;
            if(str.equals(methodInvoke))
                return;
        }
        invokeMethodSig.add(methodInvoke);
    }

    public void addMethodInvoke(List<String> methodInvoke){
        for(String str : methodInvoke){
            if(str ==null) continue;
            addMethodInvoke(str);
        }
    }

    public void addParamValue(HashMap<Integer, List<String>> invokeResult){
        if(invokeResult.toString().length() > 2000) return;
        MethodString.addValue(paramValue, invokeResult);
    }

    public void addParamValue(Integer arg, List<String> value){
        if(value.toString().length() > 2000) return;
        MethodString.addValue(paramValue, arg, value);
    }

    public void addParamValue(Integer arg, String value){
        if(value.length() > 1000) return;
        MethodString.addValue(paramValue, arg, value);
    }

    public void solve(){
        if(this.invokeMethodSig.toString().length() > 5000) {
            this.invokeMethodSig.clear();
            this.param.clear();
            this.solved = true;
            return;
        }
        if(!this.paramValue.isEmpty()){
            HashMap<Integer, List<String>> paraVa = paramValue;
            for(Integer integer : paraVa.keySet()) {
                if(paraVa.get(integer) != null)
                    this.invokeMethodSig.replaceAll(s -> s.replace("$[" + integer + "]", MethodString.getContent(paraVa.get(integer))));
            }
        }
        List<Integer> keys = new ArrayList<>(paramValue.keySet());
        this.param.removeAll(keys);
        if(param.isEmpty()) solved = true;
    }


    @Override
    public String toString() {
        if(solved || param.isEmpty()){
            return MethodString.getContent(invokeMethodSig);
        }
        StringBuilder sb = new StringBuilder("MethodParamInvoke{");
        sb.append("sootMethod=").append(sootMethod);
        sb.append(", param=").append(param);
        sb.append(", paramValue=").append(MethodString.getContent(paramValue));
        sb.append(", invokeMethodSig=");
        sb.append(MethodString.getContent(invokeMethodSig));
        sb.deleteCharAt(sb.length()-1);
        sb.append("}");
        return sb.toString();
    }

    @Override
    public boolean equals(Object o) {
        if (this == o) return true;
        if (o == null || getClass() != o.getClass()) return false;
        MethodParamInvoke that = (MethodParamInvoke) o;
        return Objects.equals(sootMethod, that.sootMethod) && Objects.equals(param, that.param) && Objects.equals(invokeMethodSig, that.invokeMethodSig);
    }

    @Override
    public int hashCode() {
        return Objects.hash(sootMethod, param, invokeMethodSig);
    }
}
