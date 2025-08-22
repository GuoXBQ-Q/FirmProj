package firmproj.main;

import firmproj.base.MethodString;
import firmproj.base.RetrofitPoint;
import firmproj.client.HttpClientFind;
import firmproj.client.RetrofitBuildFind;
import firmproj.graph.CallGraph;
import firmproj.utility.*;

import org.json.JSONArray;
import org.json.JSONObject;
import org.apache.logging.log4j.LogManager;
import org.apache.logging.log4j.Logger;

import soot.PackManager;
import soot.Scene;
import soot.SootClass;
import soot.SootMethod;
import soot.options.Options;

import org.apache.commons.cli.CommandLine;
import soot.tagkit.AnnotationTag;
import soot.tagkit.Tag;
import soot.tagkit.VisibilityAnnotationTag;
import soot.util.Chain;

import java.io.*;
import java.nio.charset.StandardCharsets;
import java.nio.file.Files;
import java.nio.file.Path;
import java.nio.file.Paths;
import java.util.*;

public class Main {
    private static final Logger LOGGER = LogManager.getLogger(Main.class);

    public static boolean outputJimpleFiles;
    private static void iniLog(String app) {
        LOGGER.info("Start Analysing {}", app);
        LOGGER.debug("Start Analysing {}", app);
        LOGGER.error("Start Analysing {}", app);
        LOGGER.warn("Start Analysing {}", app);
    }
    private static void endLog() {
        LOGGER.info("---------------------");
        LOGGER.debug("---------------------");
        LOGGER.error("---------------------");
        LOGGER.warn("---------------------");
    }

    public static void main(String[] args) throws IOException{
        CommandLine cmd = CommandLineOptions.parseOptions(CommandLineOptions.getNewOptions(), args);
        if (cmd == null || cmd.getOptionValue(CommandLineOptions.platform) == null ||
                //cmd.getOptionValue(CommandLineOptions.desc) == null ||
                (cmd.getOptionValue(CommandLineOptions.apk) == null && cmd.getOptionValue(CommandLineOptions.listOfApk) == null)) {
            LOGGER.error("cmd error");
            System.exit(1);

        }
        String outputPath = Config.RESULTDIR;

        if (cmd.hasOption(CommandLineOptions.outputPath)) {
            outputPath = cmd.getOptionValue(CommandLineOptions.outputPath);

            // if output path exist and has files, then return
            ApkContext apkContext = ApkContext.getInstance(cmd.getOptionValue(CommandLineOptions.apk));
            outputPath += apkContext.getPackageName() + ".apk/Output/";
            if (FileUtility.getOutputFile(outputPath + apkContext.getPackageName() + ".txt").exists()) {
                LOGGER.error("Output path {} already exists and is not empty", outputPath);
                System.exit(1);
            }

            QueryJson.setOutputPath(outputPath);
        }
        // set Android jar, which is later needed from soot
        Config.ANDROID_JAR_DIR = cmd.getOptionValue(CommandLineOptions.platform);
        String jarToLoad;
        if (Config.ANDROID_JAR_DIR.endsWith(".jar")) {
            jarToLoad = Config.ANDROID_JAR_DIR;
        } else {
            List<Path> paths = new ArrayList<>();
            Files.list(new File(Config.ANDROID_JAR_DIR).toPath()).forEach(paths::add);
            int max = 0;
            String currentPath = "";
            for (Path path : paths) {
                try{
                    int current = Integer.parseInt(path.getFileName().toString().replace("android-", ""));
                    if (current > max) {
                        max =current;
                        currentPath = path.toString() + "/android.jar";
                    }
                }catch (Exception e) {
                    LOGGER.error(e);
                }
            }
            jarToLoad = currentPath;

        }

        //parse description, containing methods to trace
        JSONObject targetMethods = new JSONObject();
        //set apk to analyse
        List<String> apksToAnalyse = new LinkedList<>();
        if (cmd.getOptionValue(CommandLineOptions.apk) != null) {
            apksToAnalyse.add(cmd.getOptionValue(CommandLineOptions.apk));
        } else {
            apksToAnalyse.addAll(FileUtility.getApksFromFile(cmd.getOptionValue(CommandLineOptions.listOfApk)));
        }

        // load exclusion list
        List<String> exclusionList = new ArrayList<>();
        if (cmd.getOptionValue(CommandLineOptions.exclusion) != null) {
            LOGGER.info("loading exclusion list");
            JSONObject exclusionJSON = new JSONObject(new String(Files.readAllBytes(Paths.get(cmd.getOptionValue(CommandLineOptions.exclusion)))));
            JSONArray jsonArray = exclusionJSON.getJSONArray("exclude");
            for (int i = 0; i < jsonArray.length(); i++) {
                LOGGER.info("excluded: {}", jsonArray.getString(i));
                exclusionList.add(jsonArray.getString(i));
            }
        }

        JarUtility.apkToJar(cmd.getOptionValue(CommandLineOptions.dex2jar), cmd.getOptionValue(CommandLineOptions.apk));
        //ReflectionHelper.init(cmd.getOptionValue(CommandLineOptions.apk), jarToLoad);

        for (String apk : apksToAnalyse) {
            analyzeApk(apk, exclusionList, targetMethods, outputPath, cmd);
        }
        System.exit(0);
    }
    public static void configureSoot(ApkContext apkContext, List<String> exclusionList) {
        soot.G.reset();

        Options.v().set_src_prec(Options.src_prec_apk);
        Options.v().set_process_dir(Collections.singletonList(apkContext.getAbsolutePath()));

        if (Config.ANDROID_JAR_DIR.endsWith(".jar")) {
            Options.v().set_force_android_jar(Config.ANDROID_JAR_DIR);
        } else {
            Options.v().set_android_jars(Config.ANDROID_JAR_DIR);

        }

        Options.v().set_process_multiple_dex(true);

        Options.v().set_whole_program(true);
        Options.v().set_allow_phantom_refs(true);
        //switch to output jimple
        Options.v().set_output_dir("/data/wenzhi/jimple/"+apkContext.getPackageName().replace(".","_") + "_jimple/");

        if (outputJimpleFiles) {
            Options.v().set_output_format(Options.output_format_jimple);
        } else {
            Options.v().set_output_format(Options.output_format_none);
        }
        Options.v().set_keep_line_number(true);
        //Options.v().set_keep_offset(true);
        Options.v().ignore_resolution_errors();


        //changes for exclude packages
        Options.v().set_exclude(exclusionList);
        Options.v().set_no_bodies_for_excluded(true);
        Options.v().set_ignore_resolution_errors(true);

        //soot.Main.v().autoSetOptions();

    }

    private static void initTool(String apk, List<String> exclusionList, boolean initSoot, String outputPath, CommandLine cmd) throws IOException {
        LOGGER.info("Setting up soot");
        ApkContext apkContext = ApkContext.getInstance(apk);
        if (cmd.hasOption(CommandLineOptions.dontOverwriteResult) && FileUtility.getOutputFile(outputPath).exists()) {
            System.exit(0);
        }
        if (initSoot) {
            configureSoot(apkContext, exclusionList);
        }

        LOGGER.info("Loading soot classes");
        try {
            Scene.v().loadNecessaryClasses();
        }catch (Throwable e) {
            LOGGER.error("Soot could not load classes...");
        }

        LOGGER.info("initialisation of the call graph");
        CallGraph.init();
        return;
    }

    public static void analyzeApk(String apk, List<String> exclusionList, JSONObject targetMethods, String outputPath, CommandLine cmd) throws IOException {
        //URL.setURLStreamHandlerFactory(new CustomURLStreamHandlerFactory());
        long startTime = System.currentTimeMillis();

        iniLog(apk);
        initTool(apk, exclusionList, true, outputPath, cmd);
        //Soot configuration and call graph initialisation
        long initTime = System.currentTimeMillis();

        MethodString.init();

        HashMap<SootClass, List<RetrofitPoint>> allRetrofitInterface;

        List<RetrofitPoint> allMethod = GetAllRetrofitAnnotationMethod();
        allRetrofitInterface = GetRetrofitClass(allMethod);
        List<RetrofitPoint> firmMethod = GetFirmRelatedMethod(allMethod);
        LOGGER.info("GetAllfirmMethod: " + firmMethod.size());

        HttpClientFind.findAllInterceptorClasses();
        HttpClientFind.findAllHttpClientBuildMethod();

        RetrofitBuildFind.RetrofitClassesWithMethods.putAll(allRetrofitInterface);
        RetrofitBuildFind.findAllFactoryClasses();
        RetrofitBuildFind.findAllRetrofitBuildMethod();
        RetrofitBuildFind.processPointParams(firmMethod);
        HttpClientFind.processAllClientParams();

        //SootMethod sootMethod = Scene.v().getMethod("<com.library.http.Http: java.lang.String getSign(java.lang.String)>");
        //LLMQuery.generate(sootMethod, new HashMap<>());
        //QueryJson.test();
        //List<ValuePoint> allValuePoints = getAllSolvedValuePoints(targetMethods, t, apk);
        long endTime = System.currentTimeMillis();

        writeOutPut(firmMethod, outputPath, startTime, endTime);
        writeJsonOutput(firmMethod, outputPath);
        if (Options.v().output_format() == 1) {
            PackManager.v().writeOutput();
        }

        endLog();
        //Trigger GC because old measurement data is not needed anymore
        System.gc();
        soot.G.reset();
    }

    public static HashSet<SootMethod> GetFirmRelatedMethod(HashSet<SootMethod> methods){
        HashSet<SootMethod> result = new HashSet<SootMethod>();
        for (SootMethod method: methods) {
            if(FirmwareRelated.isFirmRelated(method.getName()))
                result.add(method);
        }
        return result;
    }

    public static List<RetrofitPoint> GetFirmRelatedMethod(List<RetrofitPoint> methods){
        List<RetrofitPoint> result = new ArrayList<RetrofitPoint>();
        for (RetrofitPoint method: methods) {
            if(FirmwareRelated.isFirmRelated(method.getMethod().getName()))
                result.add(method);
        }
        return result;
    }

    public static HashMap<SootClass, List<RetrofitPoint>> GetRetrofitClass(List<RetrofitPoint> methods){
        HashMap<SootClass, List<RetrofitPoint>> result = new HashMap<>();
        for(RetrofitPoint point: methods){
            addValue(result, point.getCurrentclass(), point);
        }
        return result;
    }


    public static List<RetrofitPoint> GetAllRetrofitAnnotationMethod(){
        List<RetrofitPoint> result = new ArrayList<RetrofitPoint>();
        Chain<SootClass> classes = Scene.v().getClasses();
        for(SootClass clz: classes){
            if(!clz.isApplicationClass()) continue;
            List<SootMethod> methods = new ArrayList<>(clz.getMethods());
            for(SootMethod method: methods){
                boolean isRetrofitI = false;
                for(Tag tag: method.getTags()){
                    if(isRetrofitI) break;
                    if(tag instanceof VisibilityAnnotationTag){
                        VisibilityAnnotationTag annotationTag = (VisibilityAnnotationTag) tag;
                        for(AnnotationTag annotation: annotationTag.getAnnotations()){
                            String type = annotation.getType();
                            if(type.equals("Lretrofit2/http/GET;") || type.equals("Lretrofit2/http/POST;")) {
                                isRetrofitI = true;
                                break;
                            }
                            else{
                                //LOGGER.warn("not match type:" + type + " method: " + method);
                            }
                        }
                    }
                }
                if(isRetrofitI) {
                    RetrofitPoint m = new RetrofitPoint(method);
                    result.add(m);
                    LOGGER.info("Match New Method: " + method);
                }
            }
        }
        return result;
    }

    public static void writeOutPut(List<RetrofitPoint> methods, String outputPath, long initTime, long endTime){
        String timePath = outputPath + File.separator + "RunTime.txt";
        FileUtility.initDirs(timePath);
        File timeFile = new File(timePath);
        try (Writer writer = new BufferedWriter(new OutputStreamWriter(
                new FileOutputStream(timeFile), StandardCharsets.UTF_8))) {
            writer.write("[Time]: ");
            writer.write((endTime - initTime) + "\n");
        } catch (FileNotFoundException e) {
            throw new RuntimeException(e);
        } catch (IOException e) {
            throw new RuntimeException(e);
        }
        outputPath += ApkContext.getInstance().getPackageName() + ".txt";

        FileUtility.initDirs(outputPath);
        File outputFile = new File(outputPath);
        try (Writer writer = new BufferedWriter(new OutputStreamWriter(
                new FileOutputStream(outputFile), StandardCharsets.UTF_8))) {
            for(RetrofitPoint Rp: methods){
                LOGGER.info("now : {}", Rp.getMethod().getName());
                Rp.solve();
                writer.write("[RetrofitMethod]: \n");
                writer.write(Rp.getResult());
                LOGGER.info(Rp.getResult());
                writer.write("\n");
            }
            for(String str : HttpClientFind.finalResult){
                writer.write(str);
                writer.write("\n");
            }

            for(String str : MethodString.findUrl){
                writer.write("[Possible Url]: ");
                writer.write(str + "\n");
            }
            LOGGER.info("create:"+outputPath+"\nRealPath: "+outputFile.getAbsolutePath());

        } catch (FileNotFoundException e) {
            throw new RuntimeException(e);
        } catch (IOException e) {
            throw new RuntimeException(e);
        }
    }

    public static void writeJsonOutput(List<RetrofitPoint> methods, String outputPath){
        JSONObject jsonObject = new JSONObject();
        int i = 0;
        for(RetrofitPoint Rp: methods){
            Rp.solve();
            String result = Rp.getResult();
            jsonObject.put(String.valueOf(i), result);
            LLMQuery.checkAndGenerateJson(result);
            i++;
        }
        for(String str : HttpClientFind.finalResult){
            jsonObject.put(String.valueOf(i), str);
            LLMQuery.checkAndGenerateJson(str);
            i++;
        }

        for(String str : MethodString.findUrl){
            jsonObject.put(String.valueOf(i), "[Possible Url]: " + str);
            LLMQuery.checkAndGenerateJson(str);
            i++;
        }
        outputPath += ApkContext.getInstance().getPackageName() + ".json";
        FileUtility.initDirs(outputPath);
        try (FileWriter file = new FileWriter(outputPath)) {
            file.write(jsonObject.toString(4)); // 格式化输出
            System.out.println("结果写入到" + outputPath + "文件中！");
        }
        catch (Throwable ignore) {}
    }

    public static <K, V> void addValue(Map<K, List<V>> map, K key, V value) {
        map.computeIfAbsent(key, k -> new ArrayList<>());
        List<V> values = map.get(key);
        if(!values.contains(value)){
            values.add(value);
        }
    }

}
