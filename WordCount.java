import java.io.IOException;
import java.util.StringTokenizer;

import javax.naming.Context;

import org.apache.hadoop.conf.Configuration;
import org.apache.hadoop.fs.Path;

import org.apache.hadoop.io.IntWritable;
import org.apache.hadoop.io.LongWritable;
import org.apache.hadoop.io.Text;

import org.apache.hadoop.mapreduce.Job;
import org.apache.hadoop.mapreduce.Mapper;
import org.apache.hadoop.mapreduce.Reducer;

import org.apache.hadoop.mapreduce.lib.input.FileInputFormat;
import org.apache.hadoop.mapreduce.lib.output.FileOutputFormat;

// import org.apache.logging.log4j.*;
// import org.apache.logging.log4j.LogManager;
// import org.apache.logging.log4j.Logger;



public class WordCount {
    // private static final Logger logger = LogManager.getLogger(WordCount.class);

    public static class Map
    extends Mapper<LongWritable, Text, Text, IntWritable> {
    
        private final static IntWritable one = new IntWritable(1);
        private Text word = new Text();
    
        public void map(LongWritable key, Text value, Context context)
                throws IOException, InterruptedException {
    
            String line = value.toString().replaceAll("[^a-zA-Z ]", "").toLowerCase();
            StringTokenizer tokenizer = new StringTokenizer(line);
    
            while (tokenizer.hasMoreTokens()) {
                word.set(tokenizer.nextToken());
                context.write(word, one);  
            }
        }
    }

  public static class Reduce
       extends Reducer<Text,IntWritable,Text,IntWritable> {
    private IntWritable result = new IntWritable();

    public void reduce(Text key, Iterable<IntWritable> values, Context context)
        throws IOException, InterruptedException {

            int sum = 0;
            for (IntWritable val : values) {
                sum += val.get();
            }
            context.write(key, new IntWritable(sum));   
        }
  }

  public static void main(String[] args) throws Exception {
    long startTime = System.currentTimeMillis();

    Configuration conf = new Configuration();
    conf.setLong("mapreduce.input.fileinputformat.split.maxsize", 67108864);
    
    Job job = Job.getInstance(conf, "word count");
    job.setJarByClass(WordCount.class);
    job.setMapperClass(Map.class);
    job.setCombinerClass(Reduce.class);
    job.setReducerClass(Reduce.class);
    job.setOutputKeyClass(Text.class);
    job.setOutputValueClass(IntWritable.class);
    FileInputFormat.addInputPath(job, new Path(args[0]));
    FileOutputFormat.setOutputPath(job, new Path(args[1]));

    boolean success = job.waitForCompletion(true);
    long endTime = System.currentTimeMillis();
    // logger.info("Total Execution Time: " 
    //     + (endTime - startTime) + " milliseconds");
    // logger.info(job.waitForCompletion(true) ? 0 : 1);
    System.out.println("Total Execution Time for split size (64): " + (endTime - startTime) + " milliseconds");
    System.exit(job.waitForCompletion(true) ? 0 : 1);
  }
}