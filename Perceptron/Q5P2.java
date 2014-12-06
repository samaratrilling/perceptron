
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


public class Q5P2 {

	static HashMap<String, Double> vWeights;
	static List<String> devSentences;
	static String bestHistories;
	
	public static void main (String args[]) {
		// Initialize variables.
		vWeights = new HashMap<String, Double>();
		devSentences = new ArrayList<String>();

		try {
			// 1. Read tag model in from tag.model
			BufferedReader readTagModel = new BufferedReader(new FileReader("suffix_tagger.model"));
			String line = readTagModel.readLine();
			while (line != null) {
				storeRAndV(line);
				line = readTagModel.readLine();
			}
			readTagModel.close();
			
			// 2. Spawn a server process for each of the commands I have to use.
			String[] taggerHistoryCmd = {"python", "tagger_history_generator.py", "ENUM"};
			Process taggerHistory = spawnProcess(taggerHistoryCmd);
			String[] taggerDecoderCmd = {"python", "tagger_decoder.py", "HISTORY"};
			Process taggerDecoder = spawnProcess(taggerDecoderCmd);
			
			// 3. For each line in development data
			devSentences = readDevData("tag_dev.dat");
			
			bestHistories = perceptron(vWeights, devSentences, taggerHistory, taggerDecoder);
			printBestHistories(bestHistories, "tag_dev_suffixes.out");
			
			
			//String output = callProcess(taggerHistory, "There DET\nis VERB\nno DET\n asbestos NOUN\nin ADP\nour PRON\nproducts NOUN\nnow ADV\n. .");
					//"There DET\nis VERB\nno DET\n asbestos NOUN\nin ADP\nour PRON\nproducts NOUN\nnow ADV\n. .");
			//String yStar = callProcess(taggerDecoder, "1 * DET 23.2\n 1 * VERB 0.25\n 1 * NOUN 0.51");
		}
		catch (IOException e) {
			e.printStackTrace();
		}
		
	}
	
	public static void printBestHistories(String bestHistories, String fileOut) throws IOException{
		PrintWriter printHistories = new PrintWriter(new File(fileOut));
		printHistories.print(bestHistories);
		printHistories.close();
	}
	
	
	public static String perceptron(HashMap<String, Double> vWeights, List<String> devSentences, Process taggerHistory, Process taggerDecoder)
			throws IOException{

		// Final string to print with all the best histories.
		StringBuffer highestScoringHistories = new StringBuffer();
		
		// sentence has format: Ms\nHaag\nplays\nElianti\n.
		for (String sentence : devSentences) {
			// Will build up a file in form 1 * DET 23.2\n1 * VERB 0.25. One per sentence.
			StringBuffer historyDotScores = new StringBuffer();
			
			String[] sentenceComp = sentence.split("\n");
			
			String histories = callProcess(taggerHistory, sentence);
			String[] historyOptions = histories.split("\n");
			// historyOptions has format: 1 * DET\n1 * VERB\n...
			for (String history: historyOptions) {
				// history has format: 1 * DET
				String[] historyComp = history.split("\\s+");
				String u = historyComp[1];
				String v = historyComp[2];
				int i = Integer.parseInt(historyComp[0]);
				String ithWord = sentenceComp[i-1];
				
				// Calc feature 1
				String bigramKey = "BIGRAM:" + u + ":" + v;
				Double feature1 = vWeights.get(bigramKey);
				if (feature1 == null) {
					feature1 = 0.0;
				}
				
				// Calc feature 2
				// Get the word in the sentence corresponding to position i (arrays start at 0 and i starts at 1)
				String tagKey = "TAG:" + ithWord + ":" + v;
				Double feature2 = vWeights.get(tagKey);
				//System.out.println(tagKey + " " + weight2);
				if (feature2 == null) {
					feature2 = 0.0;
				}
				
				// Calc feature 3, suffixes.
				// SUFFIX FEATURE
				Double suffixSum = 0.0;
				ArrayList<String> suffixes = generateSuffixes(ithWord);
				for (String suffix : suffixes) {
					
					String suffixKey = "SUFFIX:" + suffix + ":" + suffix.length() + ":" + v;
					Double feature3 = vWeights.get(suffixKey);
					if (feature3 == null) {
						feature3 = 0.0;
					}
					suffixSum += feature3;
					
				}
				
				Double historyWeight = feature1 + feature2 + suffixSum;
				// historyScoreLine should be of form 1 * DET 23.2\n
				String historyScoreLine = i + " " + u + " " + v + " " + historyWeight;
				historyDotScores.append(historyScoreLine + "\n");
				//System.out.println(historyScoreLine);
				
			}// end iteration through possible histories
			// Now call tagger decoder with historyDotScores to get the best history out of all possible ones.
			// bestHistory is in format 1 * DET\n2 DET VERB\n
			String bestHistory = callProcess(taggerDecoder, historyDotScores.toString());
			
			// Match up the history with the sentence
			// bestComp has format 1 * DET
			String[] bestComp = bestHistory.split("\n");
			// Iterate through the history
			for (int b = 0; b < bestComp.length - 1; b++) {
				String bigram = bestComp[b];

				String[] bigramComp = bigram.split(" ");
				int index = Integer.parseInt(bigramComp[0]);
				String bestHistLine = sentenceComp[index-1] + " " + bigramComp[2];
				highestScoringHistories.append(bestHistLine + "\n");
			}
			highestScoringHistories.append("\n");

		}// end iteration through sentences
		return highestScoringHistories.toString();
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
	
	/**
	 * Reads in a line that will have the form TAG:asbestos:NOUN or BIGRAM:DET:NOUN.
	 * Saves every word seen in rWords; saves every line in a hashmap of featurestring -> weight.
	 * @param line
	 */
	public static void storeRAndV(String line) {
		String[] lineComp = line.split(" ");
		String[] featureStringComp = lineComp[0].split(":");

		vWeights.put(lineComp[0], Double.parseDouble(lineComp[1]));
	}
}
