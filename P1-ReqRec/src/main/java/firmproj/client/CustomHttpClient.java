package firmproj.client;

import firmproj.base.MethodString;
import org.apache.logging.log4j.LogManager;
import org.apache.logging.log4j.Logger;
import soot.SootMethod;
import soot.Unit;

import java.lang.reflect.Method;
import java.util.ArrayList;
import java.util.HashMap;
import java.util.List;
import java.util.Objects;

public class CustomHttpClient implements AbstractHttpClient {

    private static final Logger LOGGER = LogManager.getLogger(CustomHttpClient.class);

    public SootMethod sootMethod;

    public Unit unit;

    private boolean isNeedRequestContent = false;

    public List<Integer> params = new ArrayList<>();

    public HashMap<Integer, List<String>> paramValues = new HashMap<>();

    public final HashMap<String, List<String>> requestContentFromParams = new HashMap<>(); //Body, MediaType, method.

    public String result;

    public CustomHttpClient(SootMethod method, Unit unit){
        this.sootMethod = method;
        this.unit = unit;
    }

    public CustomHttpClient(){}

    public void setNeedRequestContent(boolean isNeed){
        this.isNeedRequestContent = isNeed;
    }

    public boolean isNeedRequestContent() {
        return isNeedRequestContent;
    }

    @Override
    public void setCreateUnit(Unit unit) {
        this.unit = unit;
    }

    @Override
    public void setSootMethod(SootMethod sootMethod) {
        this.sootMethod = sootMethod;
    }

    @Override
    public SootMethod getSootMethod() {
        return this.sootMethod;
    }



    @Override
    public void setParams(List<Integer> params) {
       MethodString.addValue(this.params, params);
    }

    @Override
    public List<Integer> getParams() {
        return this.params;
    }

    @Override
    public void addRequestContent(HashMap<String, List<String>> addContent) {
        MethodString.addValue(this.requestContentFromParams, addContent);
    }

    @Override
    public HashMap<String, List<String>> getRequestContent() {
        return this.requestContentFromParams;
    }

    @Override
    public void addParamValues(HashMap<Integer, List<String>> paramValues) {
        MethodString.addValue(this.paramValues, paramValues);
    }

    @Override
    public HashMap<Integer, List<String>> getParamValues() {
        return this.paramValues;
    }

    @Override
    public String getResult() {
        if(!this.paramValues.isEmpty()){
            HashMap<Integer, List<String>> paraVa = paramValues;
            for(Integer integer : paraVa.keySet()) {
                for(List<String> values : this.requestContentFromParams.values())
                    values.replaceAll(s -> s.replace("$[" + integer + "]", MethodString.getContent(paraVa.get(integer))));
            }
        }
        StringBuilder result = new StringBuilder("Client[");
        if(result.length() > 0)
            result.deleteCharAt(result.length() - 1);
        if(!requestContentFromParams.isEmpty())
            result.append(MethodString.getContent(requestContentFromParams));
        result.append("]");
        return result.toString();
    }

    @Override
    public boolean equals(Object o) {
        if (this == o) return true;
        if (o == null || getClass() != o.getClass()) return false;
        CustomHttpClient that = (CustomHttpClient) o;
        return isNeedRequestContent == that.isNeedRequestContent && Objects.equals(sootMethod, that.sootMethod) && Objects.equals(unit, that.unit) && Objects.equals(params, that.params) && Objects.equals(requestContentFromParams, that.requestContentFromParams);
    }

    @Override
    public int hashCode() {
        return Objects.hash(sootMethod, unit, isNeedRequestContent, params, requestContentFromParams);
    }

    @Override
    public String toString() {
        return "CustomHttpClient{" + sootMethod +
                ", isNeedRequestContent=" + isNeedRequestContent +
                ", params=" + params +
                ", requestContentFromParams=" + requestContentFromParams +
                '}';
    }
}
