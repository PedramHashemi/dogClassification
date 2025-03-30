# Image Classification using AWS SageMaker

Use AWS Sagemaker to train a pretrained model that can perform image classification by using the Sagemaker profiling, debugger, hyperparameter tuning and other good ML engineering practices. This can be done on either the provided dog breed classication data set or one of your choice.

## Project Set Up and Installation
Enter AWS through the gateway in the course and open SageMaker Studio. 
Download the starter files.
Download/Make the dataset available. 

## Dataset
The provided dataset is the dogbreed classification dataset which can be found in the classroom.
The project is designed to be dataset independent so if there is a dataset that is more interesting or relevant to your work, you are welcome to use it to complete the project.

In this case I have used the dog breed dataset. I have downloaded the dataset with the following command and unziped it.
```
!wget https://s3-us-west-1.amazonaws.com/udacity-aind/dog-project/dogImages.zip --no-check-certificate
!unzip dogImages.zip
```


### Access
Upload the data to an S3 bucket through the AWS Gateway so that SageMaker has access to the data. 
Using the following code I have uploaded the data in the folder dogImages folder in S3.
```
train = sagemaker_session.upload_data(
    path="dogImages/train", bucket=BUCKET, key_prefix="train"
)
test = sagemaker_session.upload_data(
    path="dogImages/test", bucket=BUCKET, key_prefix="test"
)
valid = sagemaker_session.upload_data(
    path="dogImages/valid", bucket=BUCKET, key_prefix="valid"
)
```

## Hyperparameter Tuning
What kind of model did you choose for this experiment and why? Give an overview of the types of parameters and their ranges used for the hyperparameter search

I used Resnet50, its a good model to start from since it contains a lot of information. I have added only one linear layer at the end of the model for classification.
I have tuned two hyperparameters, Learning Rate and batch_size.
Smaller batch size can somtimes yield better results but it doesn't use the memory of the machine we use.
the learning rate is also very important since the big ones can skip the local minimum and the if it's too small the speed of training is too slow but eventually might yield better results.

Remember that your README should:
- Include a screenshot of completed training jobs
  ![image](CD0387-deep-learning-topics-within-computer-vision-nlp-project-starter/Screenshot 2025-03-26 at 17.11.24.png)
- Logs metrics during the training process
  They appear in the training Job
- Tune at least two hyperparameters
- Retrieve the best best hyperparameters from all your training jobs

## Debugging and Profiling
**TODO**: Give an overview of how you performed model debugging and profiling in Sagemaker
SageMaker profiling works through a combination of:

### Amazon SageMaker Debugger: 
- Captures metrics during training.
- Framework Profiling: Collects low-level system and deep learning framework insights.
- Visualization Tools: Provides reports on CPU/GPU utilization, memory usage, data loading bottlenecks, etc.

The following are the benefits of profiling:
- Cost Optimization: Identify inefficiencies to reduce instance usage.
- Performance Tuning: Pinpoint CPU/GPU bottlenecks for faster training.
- Debugging: Catch issues related to memory leaks and inefficient code execution.
- Scalability: Helps optimize distributed training jobs.


I have used the following code to implement it:
```
rules = [
    Rule.sagemaker(rule_configs.loss_not_decreasing()),
    Rule.sagemaker(rule_configs.overfit()),
    Rule.sagemaker(rule_configs.ProfilerReport()),
]

profiler_config = ProfilerConfig(
    system_monitor_interval_millis=500,
    framework_profile_params=FrameworkProfile()
)

debugger_config = DebuggerHookConfig(
    hook_parameters={"train.save_interval": "100", "eval.save_interval": "10"}
)
```

### Results
**TODO**: What are the results/insights did you get by profiling/debugging your model?

I had two active parts regarding increasing GPU memory and memory utilization which were triggered many times.



## Model Deployment
**TODO**: Give an overview of the deployed model and instructions on how to query the endpoint with a sample input.

This was not successfull, after more than 2 months can't still make this work

**TODO** Remember to provide a screenshot of the deployed active endpoint in Sagemaker.

