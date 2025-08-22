package firmproj.base;

import firmproj.client.RetrofitBuildPoint;
import firmproj.graph.CallGraph;
import firmproj.graph.CallGraphNode;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;
import soot.*;
import soot.tagkit.*;

import java.lang.reflect.Field;
import java.lang.reflect.Method;
import java.util.*;

public class RetrofitPoint {

    private static final Logger LOGGER = LoggerFactory.getLogger(RetrofitPoint.class);
    private final SootClass Currentclass;
    private final SootMethod method;

    public RetrofitBuildPoint retrofitBuildPoint;
    public SootClass callBackClass;
    public List<SootMethod> callByMethod = new ArrayList<>();

    private final List<AnnotationTag> methodAnnotations;
    private final List<AnnotationTag> parameterAnnotations;
    private HashMap<Integer, List<String>> paramValues = new HashMap<>();
    //private final HashSet<Value> interestingVariables;


    private static List<AnnotationTag> extractMethodAnnotations(SootMethod method) {
        List<AnnotationTag> mAn = new ArrayList<>();
        for (Tag tag : method.getTags()) {
            if (tag instanceof VisibilityAnnotationTag) {
                if(((VisibilityAnnotationTag) tag).hasAnnotations())
                    mAn.addAll(((VisibilityAnnotationTag) tag).getAnnotations());
            }
        }
        return mAn;
    }

    private static List<AnnotationTag> extractParameterAnnotations(SootMethod method) {
        List<AnnotationTag> pAn = new ArrayList<>();
        for (Tag tag : method.getTags()) {
            if (tag instanceof VisibilityParameterAnnotationTag) {
                //LOGGER.info(tag.toString());
                VisibilityParameterAnnotationTag parameterAnnotationTag = (VisibilityParameterAnnotationTag) tag;
                for (VisibilityAnnotationTag vtag : parameterAnnotationTag.getVisibilityAnnotations()) {
                    if(vtag == null) break;
                    //LOGGER.info("vtag: " + vtag.toString());
                    if(vtag.hasAnnotations())
                        pAn.addAll(vtag.getAnnotations());
                }
            }
        }
        return pAn;
    }


    public RetrofitPoint(SootMethod method){
        this(method, method.getDeclaringClass());
    }

    public RetrofitPoint(SootMethod method, SootClass clas){
        this(method, clas, extractMethodAnnotations(method), extractParameterAnnotations(method));
    }

    public RetrofitPoint(SootMethod method, SootClass clas,List<AnnotationTag> mAn, List<AnnotationTag> pAn){
        this.method = method;
        this.Currentclass = clas;
        this.methodAnnotations = mAn;
        this.parameterAnnotations = pAn;
    }

    public void solve(){
        List<Integer> paramIndexs = new ArrayList<>();
        int paramIndex = -1;
        for(int i=0;i<parameterAnnotations.size();i++) {
            paramIndex++;
            paramIndexs.add(paramIndex);
            if(parameterAnnotations.get(i).toString().contains("jetbrains")) {
                paramIndexs.remove(paramIndexs.size() - 1);
                paramIndex--;
            }
        }
        HashMap<String, List<Integer>> invokeParams = new HashMap<>();
        invokeParams.put(method.getSignature(), new ArrayList<>(paramIndexs));
        int index = 0;
        for(SootMethod sootMethod : this.callByMethod) {
            if(index > 3) break;
            MethodLocal methodLocal = new MethodLocal(sootMethod, invokeParams, 2);
            methodLocal.setGetResult(true);
            methodLocal.doAnalysis();
            if(methodLocal.getInterestingParamString().containsKey(method.getSignature())) {
                HashMap<Integer, List<String>> invokeResult = MethodString.clone(methodLocal.getInterestingParamString().get(method.getSignature()));
                for(Integer integer : invokeResult.keySet()){
                    List<String> values = invokeResult.get(integer);
                    Iterator<String> iterator = values.iterator();
                    while(iterator.hasNext()){
                        String str = iterator.next();
                        if(str.startsWith("UNKNOWN") || str.isEmpty() || str.length() > 5000 || (values.size() > 10 && str.startsWith("<") && str.endsWith(">")))
                            iterator.remove();
                    }
                }
                addParamValues(invokeResult);
            }
            index++;
        }
    }

    public SootClass getCurrentclass() {
        return Currentclass;
    }

    public void addParamValues(HashMap<Integer, List<String>> paramValues){
        MethodString.addValue(this.paramValues, paramValues);
    }

    public HashMap<Integer, List<String>> getParamValues() {
        return paramValues;
    }

    public SootMethod getMethod(){
        return this.method;
    }

    public List<AnnotationTag> getMethodAnnotations() {
        return methodAnnotations;
    }

    public List<AnnotationTag> getParameterAnnotations() {
        return parameterAnnotations;
    }

    private String AnnoString(){
        StringBuilder result = new StringBuilder();
        for(AnnotationTag annotationTag : methodAnnotations){
            StringBuilder sb = new StringBuilder("[");
            if(annotationTag.toString().contains("jetbrains")) continue;
            sb.append(annotationTag.getType());
            if(!annotationTag.getElems().isEmpty()) {
                List<AnnotationElem> elems = new ArrayList<>(annotationTag.getElems());
                for(AnnotationElem elem : elems) {
                    if(elem != null) {
                        Object obj = getElemValue(elem);
                        if (obj != null)
                            sb.append(obj.toString());
                    }
                }
            }
            sb.append("],");
            result.append(sb);
        }
        if(result.length() > 0)
            result.deleteCharAt(result.length() - 1);
        if(!parameterAnnotations.isEmpty())
            result.append("\nParameter Anno = ");
        else
            result.append("\nWithout Parameter Anno;");
        int index = 0;
        for(AnnotationTag annotationTag : parameterAnnotations){
            StringBuilder sb = new StringBuilder("[");
            if(annotationTag.toString().contains("jetbrains")) continue;
            sb.append(annotationTag.getType());
            if(!annotationTag.getElems().isEmpty()) {
                List<AnnotationElem> elems = new ArrayList<>(annotationTag.getElems());
                for(AnnotationElem elem : elems) {
                    if (elem != null) {
                        Object obj = getElemValue(elem);
                        if (obj != null)
                            sb.append(obj.toString());
                        if (paramValues.containsKey(index)) {
                            sb.append("=");
                            sb.append(MethodString.getContent(paramValues.get(index)));
                        }
                    }
                }
            }
            else{
                if(paramValues.containsKey(index))
                    sb.append(MethodString.getContent(paramValues.get(index)));
            }
            sb.append("],");
            result.append(sb);
            index++;
        }
        if(result.length() > 0)
            result.deleteCharAt(result.length() - 1);
        return result.toString();
    }

    public String getResult(){
        StringBuilder result = new StringBuilder("RetrofitPoint:{\n");
        result.append("Method Name = [").append(getMethod().getSubSignature()).append("]\n");
        result.append("Method Anno = ").append(AnnoString()).append("\n");
        if(retrofitBuildPoint != null)
            result.append(retrofitBuildPoint.getResult());
        result.append("}\n");
        return result.toString();
    }

    public String toString() {
        StringBuilder result = new StringBuilder();
        result.append("RetrofitPoint: ");
        result.append(getMethod().toString());
        result.append("\nMethod Anno:");
        result.append(AnnoString());
        return result.toString();
    }

    private static Object getElemValue(AnnotationElem elem) {
        try {
            Field valueField = elem.getClass().getDeclaredField("value");
            valueField.setAccessible(true); // 允许访问私有字段
            return valueField.get(elem);
        } catch (NoSuchFieldException | IllegalAccessException e) {
            return null;
        }
    }
}
