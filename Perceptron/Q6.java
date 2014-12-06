
import java.io.BufferedReader;
import java.io.BufferedWriter;
import java.io.File;
import java.io.FileReader;
import java.io.IOException;
import java.io.InputStream;
import java.io.InputStreamReader;
import java.io.OutputStream;
import java.io.PrintWriter;
import java.lang.ProcessBuilder.Redirect;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.HashMap;
import java.util.HashSet;
import java.util.List;


public class Q6 {

	static HashMap<String, Double> vWeights;
	static List<String> devSentences;
	
	public static void main (String args[]) {
		// Initialize variables.
		vWeights = new HashMap<String, Double>();
		devSentences = new ArrayList<String>();

		try {
			
			// Spawn a server process for each of the commands I have to use.
			String[] taggerHistENUMCmd = {"python", "tagger_history_generator.py", "ENUM"};
			Process taggerHistENUM = spawnProcess(taggerHistENUMCmd);
			String[] taggerHistGOLDCmd = {"python", "tagger_history_generator.py", "GOLD"};
			Process taggerHistGOLD = spawnProcess(taggerHistGOLDCmd);
			String[] taggerDecoderCmd = {"python", "tagger_decoder.py", "HISTORY"};
			Process taggerDecoder = spawnProcess(taggerDecoderCmd);
			
			// Read in sentences from training data.
			devSentences = readDevData("tag_train.dat");
			
			long startTime = System.currentTimeMillis();
			vWeights = perceptron(vWeights, devSentences, taggerHistENUM, taggerHistGOLD, taggerDecoder, 5);
			long endTime = System.currentTimeMillis();
			System.out.println("Time taken by perceptron: " + (endTime - startTime) / 60000 + " minutes");
			printVWeights(vWeights, "experimental_tagger.model");
			
			
			//String output = callProcess(taggerHistory, "There DET\nis VERB\nno DET\n asbestos NOUN\nin ADP\nour PRON\nproducts NOUN\nnow ADV\n. .");
					//"There DET\nis VERB\nno DET\n asbestos NOUN\nin ADP\nour PRON\nproducts NOUN\nnow ADV\n. .");
			//String yStar = callProcess(taggerDecoder, "1 * DET 23.2\n 1 * VERB 0.25\n 1 * NOUN 0.51");
		}
		catch (IOException e) {
			e.printStackTrace();
		}
		
	}
	
	public static void printVWeights(HashMap<String, Double> vWeights, String outputFile) throws IOException{
		PrintWriter printV = new PrintWriter(new File(outputFile));
		for (String key : vWeights.keySet()) {
			String toPrint = key + " " + vWeights.get(key) + "\n";
			printV.print(toPrint);
		}
		printV.flush();
		printV.close();
	}
	
	public static HashMap<String, Double> perceptron(HashMap<String, Double> vWeights, List<String> devSentences,
			Process taggerENUM, Process taggerGOLD, Process taggerDecoder, int iterations)
			throws IOException{

		// Number of iterations
		for (int i = 0; i < iterations; i++) {
			System.out.println("Starting iteration " + (i+1));
			// sentence has format: Ms\nHaag\nplays\nElianti\n.
			for (String sentence : devSentences) {
				String[] words = sentence.split("\n");
				
				
				// Deal with Gold features.
				HashMap<String, Double> goldFeatureMap = new HashMap<>();
				String goldHistories = callProcess(taggerGOLD, sentence);
				// Histories format: 1 * NOUN, 2 NOUN NOUN
				String[] goldhistories = goldHistories.split("\n");
				for (String history : goldhistories) {
					String[] historyComp = history.split("\\s+");
					int index = Integer.parseInt(historyComp[0]);
					String u = historyComp[1];
					String v = historyComp[2];
					
					// BIGRAM FEATURE
					String bigramKey = "BIGRAM:" + u + ":" + v;
					if (goldFeatureMap.containsKey(bigramKey)) {
						//Increment the existing value
						Double val = goldFeatureMap.get(bigramKey);
						val += 1;
						goldFeatureMap.put(bigramKey, val);
					}
					else {
						goldFeatureMap.put(bigramKey, 1.0);
					}
					
					// TAG FEATURE
					String ithWord = words[index-1].split("\\s+")[0];
					String tagKey = "TAG:" + ithWord + ":" + v;
					if (goldFeatureMap.containsKey(tagKey)) {
						Double val = goldFeatureMap.get(tagKey);
						val += 1;
						goldFeatureMap.put(tagKey, val);
					}
					else {
						goldFeatureMap.put(tagKey, 1.0);
					}
					
					// SUFFIX FEATURE
					ArrayList<String> suffixes = generateSuffixes(ithWord);
					for (String suffix : suffixes) {
						String suffixKey = "SUFFIX:" + suffix + ":" + suffix.length() + ":" + v;
						if (goldFeatureMap.containsKey(suffixKey)) {
							Double val = goldFeatureMap.get(suffixKey);
							val += 1;
							goldFeatureMap.put(suffixKey, val);
						}
						else {
							goldFeatureMap.put(suffixKey, 1.0);
						}
					}
					
					// PREFIX FEATURE
					ArrayList<String> prefixes = generatePrefixes(ithWord);
					for (String prefix : prefixes) {
						String prefixKey = "PREFIX" + prefix + ":" + prefix.length() + ":" + v;
						if (goldFeatureMap.containsKey(prefixKey)) {
							Double val = goldFeatureMap.get(prefixKey);
							val += 1;
							goldFeatureMap.put(prefixKey, val);
						}
						else {
							goldFeatureMap.put(prefixKey, 1.0);
						}
					}
					
					// CONTAINS DASH FEATURE
					boolean hasDash = false;
					if (ithWord.contains("-")) {
						hasDash = true;
					}
					String dashKey = "DASH:" + v + ":" + hasDash;
					if (goldFeatureMap.containsKey(dashKey)) {
						Double val = goldFeatureMap.get(dashKey);
						val += 1;
						goldFeatureMap.put(dashKey, val);
					}
					else {
						goldFeatureMap.put(dashKey, 1.0);
					}

					// INITIAL CAPITAL
					boolean initCap = false;
					String[] letters = ithWord.split("");
					if (letters[0].matches("[A-Z]")) {
						initCap = true;
					}
					String capKey = "CAP:" + v + ":" + initCap;
					if (goldFeatureMap.containsKey(capKey)) {
						Double val = goldFeatureMap.get(capKey);
						val += 1;
						goldFeatureMap.put(capKey, val);
					}
					else {
						goldFeatureMap.put(capKey,  1.0);
					}
					
					// PREFIX FEATURE
				}//end iteration over gold histories
				
				// Deal with expected features.

				String enumHistories = callProcess(taggerENUM, sentence);
				String[] enumComp = enumHistories.split("\n");
				StringBuffer historyDotScores = new StringBuffer();
				for (String history : enumComp) {
					HashMap<String, Double> lineFeatureMap = new HashMap<>();
					//HashMap<String, Double> localWeights = new HashMap<>();
					String[] historyComp = history.split("\\s+");
					int index = Integer.parseInt(historyComp[0]);
					String u = historyComp[1];
					String v = historyComp[2];
					
					// BIGRAM FEATURE
					String bigramKey = "BIGRAM:" + u + ":" + v;
					Double bigramWeight = vWeights.get(bigramKey);
					if (bigramWeight == null) {
						bigramWeight = 0.0;
					}
					lineFeatureMap.put(bigramKey, bigramWeight);
					
					// TAG FEATURE
					String ithWord = words[index-1].split("\\s+")[0];
					String tagKey = "TAG:" + ithWord + ":" + v;
					Double tagWeight = vWeights.get(tagKey);
					if (tagWeight == null) {
						tagWeight = 0.0;
					}
					lineFeatureMap.put(tagKey, tagWeight);
					
					// SUFFIX FEATURE
					ArrayList<String> suffixKeys = new ArrayList<String>();
					ArrayList<String> suffixes = generateSuffixes(ithWord);
					for (String suffix : suffixes) {
						
						String suffixKey = "SUFFIX:" + suffix + ":" + suffix.length() + ":" + v;
						Double suffixWeight = vWeights.get(suffixKey);
						if (suffixWeight == null) {
							suffixWeight = 0.0;
						}
						lineFeatureMap.put(suffixKey, suffixWeight);
						suffixKeys.add(suffixKey);
						
					}// end j iteration
					
					// PREFIX FEATURE
					ArrayList<String> prefixKeys = new ArrayList<String>();
					ArrayList<String> prefixes = generatePrefixes(ithWord);
					for (String prefix : prefixes) {
						String prefixKey = "PREFIX:" + prefix + ":" + prefix.length() + ":" + v;
						Double prefixWeight = vWeights.get(prefixKey);
						if (prefixWeight == null) {
							prefixWeight = 0.0;
						}
						lineFeatureMap.put(prefixKey, prefixWeight);
						prefixKeys.add(prefixKey);
					}
					
					// CONTAINS DASH FEATURE
					boolean hasDash = false;
					if (ithWord.contains("-")) {
						hasDash = true;
					}
					String dashKey = "DASH:" + v + ":" + hasDash;
					Double dashWeight = vWeights.get(dashKey);
					if (dashWeight == null) {
						dashWeight = 0.0;
					}
					lineFeatureMap.put(dashKey, dashWeight);
					
					// INITAL CAPITAL FEATURE
					boolean initCap = false;
					String[] letters = ithWord.split("");
					if (letters[0].matches("[A-Z]")) {
						initCap = true;
					}
					String capKey = "CAP:" + v + ":" + initCap;
					Double capWeight = vWeights.get(capKey);
					if (capWeight == null) {
						capWeight = 0.0;
					}
					lineFeatureMap.put(capKey, capWeight);
					
					Double score = lineFeatureMap.get(bigramKey) +
							 lineFeatureMap.get(tagKey) + lineFeatureMap.get(dashKey) + lineFeatureMap.get(capKey);
					
					for (String fKey : suffixKeys) {
						Double sufWeight = lineFeatureMap.get(fKey);	
						score += sufWeight;
					}

					String historyLine = index + " " + u + " " + v + " " + score;
					historyDotScores.append(historyLine + "\n");
				}// end of iteration over enum histories
				// Formal for best-scoring history: 1 * DET\n2 DET VERB
				String bestScoringHistory = callProcess(taggerDecoder, historyDotScores.toString());
				String[] bestTagging = bestScoringHistory.split("\n");
				
				// Check whether or not the best history is the same as the gold history.
				String[] goldHistory = removeSpaces(goldhistories);
				String[] bestTagNoSTOP = removeStop(bestTagging);
				boolean goldEqualsExpected = arraysAreEqual(bestTagNoSTOP, goldHistory);
				
				if (!goldEqualsExpected) {
					// Get back the line feature map for the best scoring expected sentence.

					for (String history : bestTagNoSTOP) {
						HashMap<String, Double> taggingFeatureMap = new HashMap<>();
						String[] historyComp = history.split("\\s+");
						int index = Integer.parseInt(historyComp[0]);
						String u = historyComp[1];
						String v = historyComp[2];
						
						// BIGRAM FEATURE
						String bigramKey = "BIGRAM:" + u + ":" + v;
						taggingFeatureMap.put(bigramKey, 1.0);
						
						// TAG FEATURE
						String ithWord = words[index-1].split("\\s+")[0];
						String tagKey = "TAG:" + ithWord + ":" + v;
						taggingFeatureMap.put(tagKey, 1.0);
						
						// SUFFIX FEATURE
						ArrayList<String> suffixes = generateSuffixes(ithWord);
						for (String suffix : suffixes) {	
							String suffixKey = "SUFFIX:" + suffix + ":" + suffix.length() + ":" + v;
							taggingFeatureMap.put(suffixKey, 1.0);
						}
						
						// PREFIX FEATURE
						ArrayList<String> prefixes = generatePrefixes(ithWord);
						for (String prefix : prefixes) {
							String prefixKey = "PREFIX:" + prefix + ":" + prefix.length() + ":" + v;
							taggingFeatureMap.put(prefixKey, 1.0);
						}
						
						// DASH FEATURE
						boolean hasDash = false;
						if (ithWord.contains("-")) {
							hasDash = true;
						}
						String dashKey = "DASH:" + v + ":" + hasDash;
						taggingFeatureMap.put(dashKey, 1.0);
						
						// CAP FEATURE
						boolean initCap = false;
						String[] letters = ithWord.split("");
						if (letters[0].matches("[A-Z]")) {
							initCap = true;
						}
						String capKey = "CAP:" + v + ":" + initCap;
						taggingFeatureMap.put(capKey, 1.0);
						
						// Update v for expected tagging.
						for (String expKey : taggingFeatureMap.keySet()) {
							Double value = vWeights.get(expKey);
							if (value == null) {
								value = 0.0;
							}
							
							// If the key is already in v, subtract the value of the expected weight.
							// Otherwise, just store 0 - the value of the expected weight.
							Double toSubtract = taggingFeatureMap.get(expKey);
							value -= toSubtract;

							vWeights.put(expKey, value);
						}
			
					}// end iteration through the full best tagging.
					
					// Update v
					// For gold keys
					for (String goldKey : goldFeatureMap.keySet()) {
						Double value = vWeights.get(goldKey);
						if (value == null) {
							value = 0.0;
						}
						// If the key is already in v, add the value of the gold weight.
						// Otherwise, just store the value of the gold weight.
						Double toAdd = goldFeatureMap.get(goldKey);
						value += toAdd;

						vWeights.put(goldKey, value);
					}

				} // end check for gold = expected
				
			}// end iteration through sentences
			
			System.out.println("Finished iteration " + (i + 1));
		}// end this iteration of the algorithm
		return vWeights;
	}
	
	public static String[] removeStop (String[] array) {
		String[] toReturn = new String[array.length-1];
		for (int i = 0; i < array.length - 1; i++) {
			toReturn[i] = array[i];
		}
		return toReturn;
	}
	
	public static boolean arraysAreEqual(String[] array1, String[] array2) {
		if (array1.length != array2.length) {
			return false;
		}
		for (int i = 0; i < array1.length; i++) {
			if (array1[i].equals(array2[i])) {
				continue;
			}
			else {
				return false;
			}
		}
		return true;
	}
	
	public static String[] removeSpaces(String[] goldTagging) {
		String[] finalTagging = new String[goldTagging.length];
		for (int i = 0; i < goldTagging.length; i++) {
			String s = goldTagging[i];
		
			String[] words = s.split("\\s+");
			StringBuffer hist = new StringBuffer();
			for (String w : words) {
				hist.append(w + " ");
			}
			String finalHist = hist.toString().trim();
			finalTagging[i] = finalHist;
		}
		return finalTagging;
	}
	
	public static ArrayList<String> generateSuffixes(String word) {
		ArrayList<String> suffixes = new ArrayList<String>();
		for (int length = 1; length < 4; length ++) {
			StringBuffer suffix = new StringBuffer();
			int currentPos = word.length() - 1;
			for (int j = 0; j < length; j++) {
				suffix.append(word.charAt(currentPos));
				currentPos --;
				if (currentPos < 0) {
					break;
				}
			}
			suffixes.add(suffix.reverse().toString());
			if (currentPos < 0) {
				break;
			}
		}
		return suffixes;
	}
	
	public static ArrayList<String> generatePrefixes (String word) {
		ArrayList<String> prefixes = new ArrayList<String>();
		for (int length = 3; length < 5; length ++) {
			StringBuffer prefix = new StringBuffer();
			int currentPos = 0;
			for (int j = 0; j < length; j++) {
				prefix.append(word.charAt(currentPos));
				currentPos++;
				if (currentPos >= word.length()) {
					break;
				}
			}
			prefixes.add(prefix.toString());
			if (currentPos >= word.length()) {
				break;
			}
		}
		return prefixes;
	}
	
	public static Process spawnProcess(String[] command) throws IOException{
		List<String> commandList = new ArrayList<String>(Arrays.asList(command));
		ProcessBuilder process = new ProcessBuilder(commandList);
		Process p = process.start();
		assert process.redirectInput() == Redirect.PIPE;
		assert process.redirectOutput() == Redirect.PIPE;
		return p;
	}
	
	public static String callProcess(Process p, String arg) throws IOException{
		OutputStream stdin = p.getOutputStream();
		PrintWriter stdinWriter = new PrintWriter(stdin);
		InputStream stdout = p.getInputStream();
		BufferedReader stdoutReader = new BufferedReader(new InputStreamReader(stdout));
		
		stdinWriter.write(arg + "\n");
		stdinWriter.flush();
		
		StringBuffer output = new StringBuffer();
		String line = stdoutReader.readLine();
		while(!line.equals("")) {
			output.append(line + "\n");
			line = stdoutReader.readLine();
		}
		return output.toString();

	}
	
	public static ArrayList<String> readDevData(String devFile) throws IOException {
		BufferedReader devReader = new BufferedReader(new FileReader(devFile));
		ArrayList<String> sentences = new ArrayList<String>();
		StringBuffer sentence = new StringBuffer();

		String line = devReader.readLine();
		while (line != null) {
			if (line.equals("")) {
				sentences.add(sentence.toString());
				sentence = new StringBuffer();
			}
			else {
				sentence.append(line + "\n");
			}
			line = devReader.readLine();
		}
		return sentences;
	}
	
}
