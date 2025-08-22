package firmproj.utility;

import java.util.ArrayList;
import java.util.HashMap;
import java.util.HashSet;
import java.util.List;
import java.util.regex.Pattern;
import java.util.regex.Matcher;
public class FirmwareRelated {
    private static final String FIRM_RELATED_MATCHER = "(?i).*(Firmware|downloadFIrmware|FirmwareUpdate|needOta|CheckOta|needUpdate|firware|fireware|getdeviceinfo|latestfirmware|checkUpdate|fw_update|checklatest|fwurl|swversion|checkIsOTAUpdateNeeded|otaUpdateRequired|fwinfo|newfwinfo|deviceOTA|deviceupdate|checknew|update|newVersion|version|upgrade|getIP|domain|getServer|Region|latestfw|checkVersion|serverzone).*";
    private static final Pattern FIRM_RELATED_PATTERN = Pattern.compile(FIRM_RELATED_MATCHER);

    public static boolean isFirmRelated(String methodString){
        Matcher matcher = FIRM_RELATED_PATTERN.matcher(methodString);
        return matcher.find();
    }

    public static HashSet<String> matchFields(String input) {
        HashSet<String> fields = new HashSet<>();
        String regex = "<([a-zA-Z0-9._$]+):\\s+([a-zA-Z0-9._$]+)\\s+([a-zA-Z_][a-zA-Z0-9_]*)>";

        Pattern pattern = Pattern.compile(regex);
        Matcher matcher = pattern.matcher(input);

        while (matcher.find()) {
            // 将匹配的结果添加到列表中
            fields.add(matcher.group());
        }

        return fields;  // 如果没有匹配到，返回空的列表
    }

    public static String matchMethod(String input){
        String regex = "<\\s*([a-zA-Z_][\\w.$]*)\\s*:\\s*([a-zA-Z_][\\w.$]*)\\s+([a-zA-Z_]\\w*)\\s*\\(([^)]*)\\)\\s*>";

        Pattern pattern = Pattern.compile(regex);
        Matcher matcher = pattern.matcher(input);

        if (matcher.matches()){
            return matcher.group(0);
        }
        else
            return "";
    }

    public static HashMap<String, HashMap<Integer, List<String>>> matchInvokeSig(String input){

        // Updated regex to include the additional part at the end
        String regex = "(<\\s*[a-zA-Z_][\\w.$]*\\s*:\\s*[a-zA-Z_][\\w\\[\\].$]*\\s+[a-zA-Z_]\\w*\\s*\\([^)]*\\)\\s*>)(\\s*\\{.*})$";

        HashMap<String, HashMap<Integer, List<String>>> resultMap = new HashMap<>();

        Pattern pattern = Pattern.compile(regex);
        Matcher matcher = pattern.matcher(input);

        while (matcher.find()) {

            String methodSignature = matcher.group(1).trim();

            String braceContent = matcher.group(2).trim();

            resultMap.put(methodSignature, new HashMap<>());

            HashMap<Integer, List<String>> valueMap = parseBraceContent(braceContent);

            if(!resultMap.get(methodSignature).isEmpty())
                continue;
            resultMap.put(methodSignature, valueMap);
            //System.out.println("Method Signature: " + methodSignature);
            //System.out.println("Parsed Values: " + resultMap);

        }
        return resultMap;
    }


    private static HashMap<Integer, List<String>> parseBraceContent(String braceContent) {
        HashMap<Integer, List<String>> valueMap = new HashMap<>();
        braceContent = braceContent.substring(1, braceContent.length() - 1).trim(); // Remove the enclosing {}

        int braceLevel = 0;
        int bracketLevel = 0;
        StringBuilder currentKey = new StringBuilder();
        StringBuilder currentValue = new StringBuilder();
        boolean isParsingValue = false;
        List<String> currentList = null;
        Integer currentKeyInt = null;

        try {
            for (int i = 0; i < braceContent.length(); i++) {
                char c = braceContent.charAt(i);

                if (c == '{') {
                    braceLevel++;
                } else if (c == '}') {
                    braceLevel--;
                } else if (c == '[') {
                    bracketLevel++;
                } else if (c == ']') {
                    bracketLevel--;
                }

                if (braceLevel == 0 && bracketLevel == 0 && c == ',') {
                    if (isParsingValue) {
                        currentList.add(currentValue.toString().trim());
                        valueMap.put(currentKeyInt, currentList);
                        isParsingValue = false;
                        currentValue.setLength(0);
                    }
                    currentKey.setLength(0);
                    currentList = null;
                } else if (!isParsingValue && c == '=') {
                    currentKeyInt = Integer.parseInt(currentKey.toString().trim());
                    currentList = new ArrayList<>();
                    isParsingValue = true;
                } else {
                    if (isParsingValue) {
                        currentValue.append(c);
                    } else {
                        currentKey.append(c);
                    }
                }
            }
        }catch (Exception ignored){}

        if (isParsingValue) {
            currentList.add(currentValue.toString().trim());
            valueMap.put(currentKeyInt, currentList);
        }

        return valueMap;
    }
}
