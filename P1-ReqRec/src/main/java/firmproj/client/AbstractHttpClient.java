package firmproj.client;

import soot.SootMethod;
import soot.Unit;

import java.util.HashMap;
import java.util.List;

public interface AbstractHttpClient {
    void setNeedRequestContent(boolean isNeed);

    boolean isNeedRequestContent();

    void setCreateUnit(Unit unit);

    void setSootMethod(SootMethod sootMethod);

    SootMethod getSootMethod();

    void setParams(List<Integer> params);

    List<Integer> getParams();

    void addRequestContent(HashMap<String, List<String>> addContent);

    HashMap<String, List<String>> getRequestContent();

    void addParamValues(HashMap<Integer, List<String>> paramValues);

    HashMap<Integer, List<String>> getParamValues();

    String getResult();

}
