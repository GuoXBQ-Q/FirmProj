package firmproj.client;

import firmproj.base.MethodString;
import soot.SootMethod;
import soot.Unit;

import java.util.ArrayList;
import java.util.List;
import java.util.Objects;

public class RetrofitBuildPoint {
    public List<String> baseUrl = new ArrayList<>();
    private final List<ConverterFactory> converterFactory = new ArrayList<>();
    private final List<CallFactory> callFactory = new ArrayList<>();
    private final List<okHttpClient> okHttpClients = new ArrayList<>();

    private SootMethod currentMethod;
    private Unit createUnit;
    private String createClass;

    public boolean classFromParam = false;
    public boolean urlFromParam = false;

    public List<Integer> classParam = new ArrayList<>();
    public List<Integer> urlParam = new ArrayList<>();

    public RetrofitBuildPoint(Unit unit){
        this.createUnit = unit;
    }

    public RetrofitBuildPoint(RetrofitBuildPoint old){
        this.baseUrl.addAll(old.baseUrl);
        this.converterFactory.addAll(old.converterFactory);
        this.callFactory.addAll(old.callFactory);
        this.okHttpClients.addAll(old.okHttpClients);
        this.createClass = old.createClass;
        this.createUnit = old.createUnit;
        this.currentMethod = old.currentMethod;
    }

    public RetrofitBuildPoint(SootMethod method, Unit unit){
        this(unit);
        currentMethod = method;
    }

    public Unit getCreateUnit(){
        return this.createUnit;
    }

    public List<String> getBaseUrl() {
        return baseUrl;
    }

    public SootMethod getCurrentMethod() {
        return currentMethod;
    }

    public String getCreateClass() {
        return createClass;
    }

    public void setCurrentMethod(SootMethod currentMethod) {
        this.currentMethod = currentMethod;
    }

    public void setCreateUnit(Unit createUnit) {
        this.createUnit = createUnit;
    }

    public void setBaseUrl(String url){
        if(!this.baseUrl.contains(url))
            this.baseUrl.add(url);
    }

    public void setBaseUrl(List<String> urls){
        for(String url : urls) {
            if (!this.baseUrl.contains(url))
                this.baseUrl.add(url);
        }
    }

    public List<okHttpClient> getOkHttpClients() {
        return okHttpClients;
    }

    public void setOkHttpClients(List<AbstractHttpClient> clients){
        for(AbstractHttpClient client: clients){
            okHttpClient okClient = (okHttpClient) client;
            if(!okHttpClients.contains(okClient))
                okHttpClients.add(okClient);
        }
    }

    public void setOkHttpClient(okHttpClient client){
        if(!okHttpClients.contains(client))
            okHttpClients.add(client);

    }

    public void addCallFactory(AbstractFactory abstractFactory){
        CallFactory callFactory = (CallFactory) abstractFactory;
        if(this.callFactory.contains(callFactory)) return;
        this.callFactory.add(callFactory);
    }

    public void addConverterFactory(AbstractFactory abstractFactory){
        ConverterFactory converterFactory = (ConverterFactory) abstractFactory;
        if(this.converterFactory.contains(converterFactory)) return;
        this.converterFactory.add(converterFactory);
    }

    public List<CallFactory> getCallFactory() {
        return callFactory;
    }

    public List<ConverterFactory> getConverterFactory() {
        return converterFactory;
    }

    public void setCreateClass(String createClass) {
        this.createClass = createClass;
    }

    @Override
    public boolean equals(Object o) {
        if (this == o) return true;
        if (o == null || getClass() != o.getClass()) return false;
        RetrofitBuildPoint that = (RetrofitBuildPoint) o;
        return classFromParam == that.classFromParam && urlFromParam == that.urlFromParam && Objects.equals(baseUrl, that.baseUrl) && Objects.equals(converterFactory, that.converterFactory) && Objects.equals(callFactory, that.callFactory) && Objects.equals(okHttpClients, that.okHttpClients) && Objects.equals(currentMethod, that.currentMethod) && Objects.equals(createClass, that.createClass) && Objects.equals(classParam, that.classParam) && Objects.equals(urlParam, that.urlParam);
    }

    @Override
    public int hashCode() {
        return Objects.hash(baseUrl, converterFactory, callFactory, okHttpClients, currentMethod, createClass, classFromParam, urlFromParam, classParam, urlParam);
    }

    public String getResult(){
        StringBuilder result = new StringBuilder("BuildPoint=[");
        result.append("baseUrl=").append(MethodString.getContent(baseUrl)).append(",");
        if(!callFactory.isEmpty()) {
            result.append("CallFactory=[");
            for (CallFactory callFactory1 : callFactory)
                result.append(callFactory1.getResult()).append(",");
            if(result.toString().endsWith(","))
                result.deleteCharAt(result.length() - 1);
            result.append("]");
        }

        if(!converterFactory.isEmpty()) {
            result.append("ConverterFactory=[");
            for (ConverterFactory converterFactory1 : converterFactory)
                result.append(converterFactory1.getResult()).append(",");
            if(result.toString().endsWith(","))
                result.deleteCharAt(result.length() - 1);
            result.append("]");
        }

        for(okHttpClient okHttpClient : okHttpClients){
            result.append(okHttpClient.getResult()).append(",");
        }

        if(result.length() > 0)
            result.deleteCharAt(result.length() - 1);
        result.append("]");
        return result.toString();
    }

    @Override
    public String toString() {
        return "---RetrofitBuildPoint{" +
                "currentMethod=" + currentMethod +
                ", baseUrl=" + MethodString.getContent(baseUrl) +
                ", createClass='" + createClass + '\'' +
                ", urlFromParam=" + urlFromParam +
                ", urlParam=" + urlParam +
                ", classFromParam=" + classFromParam +
                ", classParam=" + classParam +
                ", createUnit=" + createUnit +
                ", converterFactory=" + MethodString.getContent(converterFactory) +
                ", callFactory=" + MethodString.getContent(callFactory) +
                ", okHttpClients=" + MethodString.getContent(okHttpClients) +
                "}---";
    }

}
