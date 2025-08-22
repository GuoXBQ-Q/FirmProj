package firmproj.client;


import firmproj.utility.LLMQuery;
import firmproj.utility.QueryJson;
import soot.SootClass;
import soot.SootMethod;

import java.util.HashMap;
import java.util.Objects;

public class CallFactory implements AbstractFactory{
    public SootClass currentClass;
    public SootMethod currentMethod;
    public QueryJson queryJson;

    public CallFactory(SootClass clz){
        this.currentClass = clz;
    }

    public void init(){
        for(SootMethod sootMethod: currentClass.getMethods()){
            if(sootMethod.getName().contains("newCall")){
                this.currentMethod = sootMethod;
            }
        }
        if(this.currentMethod != null){
            QueryJson queryJson1 = LLMQuery.generateHttp(currentMethod);
            if(queryJson1.getTargetMethodSubsSig().contains(currentMethod.getName())){
                this.queryJson = queryJson1;
            }
        }
    }

    @Override
    public boolean equals(Object o) {
        if (this == o) return true;
        if (o == null || getClass() != o.getClass()) return false;
        CallFactory that = (CallFactory) o;
        return Objects.equals(currentClass, that.currentClass) && Objects.equals(currentMethod, that.currentMethod);
    }

    @Override
    public int hashCode() {
        return Objects.hash(currentClass, currentMethod);
    }

    @Override
    public void generateQuery() {
        this.queryJson.doGenerate();
    }

    public String getResult(){
        String result = this.currentClass.getName();
        generateQuery();
        return result;
    }

    @Override
    public String toString() {
        return "CallFactory{" +
                "currentClass=" + currentClass +
                ", currentMethod=" + currentMethod +
                '}';
    }
}
