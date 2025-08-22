package firmproj.objectSim;

import firmproj.base.ValueContext;
import soot.SootClass;
import soot.SootMethod;

import java.util.HashMap;
import java.util.List;

public interface AbstractClz {

    void solve();

    void init();

    void addValueContexts(ValueContext valueContext);

    SootMethod getParentMethod();

    SootClass getCurrentClass();

    List<ValueContext> getCurrentValues();

    Object getResult();

    boolean isSolved();

    @Override
    String toString();

}
