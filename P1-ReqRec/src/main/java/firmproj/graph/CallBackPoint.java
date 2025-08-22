package firmproj.graph;

import firmproj.base.RetrofitPoint;
import soot.SootClass;
import soot.SootField;
import soot.SootMethod;
import soot.Unit;

import java.util.ArrayList;
import java.util.List;

public class CallBackPoint {
    public SootClass callBackClass;
    public List<SootMethod> parentMethod = new ArrayList<>();
    public List<SootMethod> relatedPoint = new ArrayList<>();
    public final List<String> relatedFields = new ArrayList<>();

    public CallBackPoint(SootClass sootClass){
        this.callBackClass = sootClass;
    }

    public void addField(SootField field){
        String fieldStr = field.toString();
        if(!relatedFields.contains(fieldStr))
            relatedFields.add(fieldStr);
    }

    public void addField(List<SootField> fields){
        for(SootField field : fields)
            addField(field);
    }
}
