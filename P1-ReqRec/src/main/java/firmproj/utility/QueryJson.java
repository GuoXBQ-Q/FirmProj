package firmproj.utility;

import firmproj.main.ApkContext;
import org.apache.logging.log4j.LogManager;
import org.apache.logging.log4j.Logger;
import soot.*;
import org.json.JSONObject;

import java.io.File;
import java.io.FileWriter;
import java.io.IOException;
import java.util.*;

public class QueryJson {
    private static final Logger LOGGER = LogManager.getLogger(QueryJson.class);
    public static String outputPath;
    public SootMethod targetMethod;
    private String targetClass;
    private String targetMethodSubsSig;
    private List<List<String>> parameterValues = new ArrayList<>();
    private List<String> relatedMethodsSig = new ArrayList<>();
    private boolean isHttp = false;

    public QueryJson(String className, String methodSubsSig, List<List<String>> parameters, List<String> relatedMethodsSig){
        this.targetClass = className;
        this.targetMethodSubsSig = methodSubsSig;
        this.parameterValues = parameters;
        this.relatedMethodsSig = relatedMethodsSig;
    }

    public QueryJson(){}

    public static void setOutputPath(String path){
        outputPath = path;
    }

    public boolean doGenerate(){
        return GenerateJson(targetClass, targetMethodSubsSig, parameterValues, relatedMethodsSig, isHttp);
    }

    public static void test(){
        String clsName = "com.gooclient.anycam.utils.ShareDataUtl";
        String methodSubSig = "java.lang.String decodeShareJson(java.lang.String)";
        List<List<String>> params= new ArrayList<>(List.of(List.of("clipData")));
        List<String> otherMethods = new ArrayList<>(List.of("<com.gooclient.anycam.utils.RC4_Base64_encode_decode: java.lang.String decode3(java.lang.String,java.lang.String)>","<com.gooclient.anycam.utils.Base64: byte[] decode(byte[])>","<com.gooclient.anycam.utils.RC4Test: byte[] GetKey(byte[],int)>","<com.gooclient.anycam.utils.RC4Test: byte[] RC4(byte[],java.lang.String)>"));

        GenerateJson(clsName, methodSubSig, params, otherMethods, false);
    }

    public static boolean GenerateJson(String targetClassName, String targetMethodSubSIg, List<List<String>> parameter, List<String> relatedMethods, boolean isHttp) {
        // 输出JSON字符串到文件
        String fileName = outputPath + "LLM-Query/";

        if(isHttp){
            fileName = fileName + "http_";
        }
        fileName = fileName + "query_<" +
                targetClassName +
                ": " +
                targetMethodSubSIg +
                ">.json";

        FileUtility.initDirs(fileName);
        File check = new File(fileName);
        if(check.exists()){
            return false;
        }

        List<List<String>> parameters = new ArrayList<>(parameter);
        List<String> relatedMethodNames = new ArrayList<>(relatedMethods);
        // 添加相关方法名，例如：
        // relatedMethodNames.add("relatedMethod1");
        // relatedMethodNames.add("relatedMethod2");

        // 获取目标类
        SootClass targetClass = Scene.v().loadClassAndSupport(targetClassName);
        targetClass.setApplicationClass();
        for(SootMethod method: targetClass.getMethods()){
            //System.out.println(method.getSubSignature());
        }

        // 找到目标方法
        SootMethod targetMethod = targetClass.getMethod(targetMethodSubSIg);

        // 获取目标方法所有的语句
        List<String> targetStatements = new ArrayList<>();
        Map<String, String> relatedMethodsCode = new HashMap<>();
        String targetCode = "";
        try {
            // 获取目标方法的Body
            Body targetBody = targetMethod.retrieveActiveBody();
            for (Unit unit : targetBody.getUnits()) {
                targetStatements.add(unit.toString() + ";");
            }

        targetCode = String.join(" ", targetStatements);

        // 获取相关方法的代码

        for (String methodName : relatedMethodNames) {
            String className = parseClassNameFromSignature(methodName);
            String subSignature = parseSubSignatureFromSignature(methodName);
            //System.out.println(subSignature);

            SootClass otherClass = Scene.v().loadClassAndSupport(className);
            SootMethod method = otherClass.getMethod(subSignature);
            if(!otherClass.isApplicationClass() || !method.isConcrete()) continue;
            Body methodBody = method.retrieveActiveBody();
            List<String> methodStatements = new ArrayList<>();
            for (Unit unit : methodBody.getUnits()) {
                methodStatements.add(unit.toString() + ";");
            }
            relatedMethodsCode.put(methodName, String.join(" ", methodStatements));
        }
        }
        catch (Exception ignore){}

        // 构建JSON对象
        JSONObject jsonObject = new JSONObject();
        JSONObject targetMethodObject = new JSONObject();

        targetMethodObject.put("Body", targetCode);
        targetMethodObject.put("Parameters", parameters);  // 可以为外部传入的参数列表

        JSONObject relatedMethodsObject = new JSONObject();
        for (Map.Entry<String, String> entry : relatedMethodsCode.entrySet()) {
            relatedMethodsObject.put(entry.getKey(), entry.getValue());
        }
        targetMethodObject.put("Related Methods", relatedMethodsObject);

        jsonObject.put(targetMethod.getSignature(), targetMethodObject);

        try (FileWriter file = new FileWriter(fileName)) {
                file.write(jsonObject.toString(4)); // 格式化输出
                System.out.println("JSON内容已成功写入到" + fileName + "文件中！");
                return true;
            } catch (Throwable ignore) { return false;
        }

    }

    public void setTargetClass(String targetClass) {
        this.targetClass = targetClass;
    }

    public void setTargetMethodSubsSig(String targetMethodSubsSig) {
        this.targetMethodSubsSig = targetMethodSubsSig;
    }

    public void addParameters(List<String> params){
        this.parameterValues.add(params);
    }

    public void setHttp(boolean http) {
        isHttp = http;
    }

    public boolean isHttp() {
        return isHttp;
    }

    public void setParameterValues(List<List<String>> parameterValues) {
        this.parameterValues = parameterValues;
    }

    public void setRelatedMethodsSig(List<String> relatedMethodsSig) {
        this.relatedMethodsSig = relatedMethodsSig;
    }

    public void addRelatedMethodSig(String relatedMethodSig){
        if(!this.relatedMethodsSig.contains(relatedMethodSig))
            this.relatedMethodsSig.add(relatedMethodSig);
    }

    public String getTargetMethodSubsSig() {
        return targetMethodSubsSig;
    }

    private static String parseClassNameFromSignature(String signature) {
        int start = signature.indexOf('<') + 1;
        int end = signature.indexOf(':');
        return signature.substring(start, end);
    }

    private static String parseSubSignatureFromSignature(String signature) {
        int start = signature.indexOf(':') + 2;
        int end = signature.indexOf('>');
        return signature.substring(start, end);
    }

    @Override
    public boolean equals(Object o) {
        if (this == o) return true;
        if (o == null || getClass() != o.getClass()) return false;
        QueryJson queryJson = (QueryJson) o;
        return Objects.equals(targetClass, queryJson.targetClass) && Objects.equals(targetMethodSubsSig, queryJson.targetMethodSubsSig);
    }

    @Override
    public int hashCode() {
        return Objects.hash(targetClass, targetMethodSubsSig);
    }
}