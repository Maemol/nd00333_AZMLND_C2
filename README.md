# Udacity Project : Operationalizing Machine Learning

In this project, we are going to explore Azure AutoML from training to inference and the use of a Pipeline to encapsulate an autoML run.

The dataset we are using is a dataset from a bank marketing campaign. You can find more information here : https://archive.ics.uci.edu/ml/datasets/Bank+Marketing

This a classification problem, the target feature is the column 'y'. It's a binary classification (Yes or No).

There are 10 input features :
![Dataset preview](/img/Dataset_preview.png)

## Architectural Diagram

The architecture we are going to use to train and deploy a model that can predict if a customer is going to buy the product or not can be describe as follow:

![AutoML Architecture for this project](/img/AutoML_Architecture.png)

If we zoom inside the AutoML part, there is a nice summary diagram in the Azure AutoML documentation : https://docs.microsoft.com/en-us/azure/machine-learning/concept-automated-ml 

![AutoML process](/img/automl-concept-diagram.png)


## Key Steps

Let's look at this project step by step.

### Dataset

First we have to register a dataset in order to use it in the AutoML run.

![Registered Datasets](/img/Registered_Datasets.png)

### Compute

The compute cluster created for this project was a compute of 6 DS12 nodes (CPU only, low priority). The goal was to have access to 5 concurrent interations so the cluster need to have more than 5 nodes.

![Compute cluster](/img/Cluster.png)


### AutoML

Now that we have a Dataset registered and a Compute created, we can start to configure an AutoML run. We set the task to Classification and the Primary metric to optimize to AUC_Weighted (because of the dataset suffer from an imbalance problem).

We set a max concurrent run to 5 and an experiment timeout to 20 minutes.

![Experiment completed](/img/Experiment_completed.png)

After 23 minutes the experiment finished and we can now have a look at the best model found during the run.


![Experiment details](/img/Experiment_details.png)


The best model used a VotingEnsemble algorithm wich gave a value of 0.94751 for the AUC_weighted metric. It was the 67th run that found the best results.

![Best Model](/img/Best_model.png)


We can have a look at the other model tested.

![Models_list](/img/Models_list.png)


### Deployment of the best model

Now that we have found a model with a good performance, we want to deploy it in order to use this model to make prediction via a rest Endpoint.

When we deploy the model, the model is registered in Azure ML and an endpoint is created. Here we chose a Azure Container Instance (ACI). By default App Insights is disabled.


![Endpoint creation](/img/AppInsights_False.png)

We can enable App Insights from the SDK after the creation of the endpoint.

    from azureml.core import Workspace
    from azureml.core.webservice import Webservice
    
    # Requires the config to be downloaded first to the current working directory
    ws = Workspace.from_config()
    
    # Set with the deployment name
    name = "bank-model-deploy"
    
    # load existing web service
    service = Webservice(name=name, workspace=ws)
    
    # enable application insight
    service.update(enable_app_insights=True)
    
    logs = service.get_logs()

Then we run logs.py

![Logs enabled](/img/Logs.png)

And we can now see that AppInsights is enabled and that the endpoint is healthy.

![Endpoint healthy](/img/AppInsights_True.png)


Once the endpoint is created, we can use the swagger URI provided by Azure ML during the creation of the endpoint to have a look at the Post /score method in order to know how to make prediction.

![Swagger](/img/Swagger.png)

Swagger documentation give us information about the json input format. We can now make a call to our endpoint to test if everything work fine.

We are going to use a python script that create a Json string from a dict with all the required input:


    data = {"data":
            [
              {
                "age": 17,
                "campaign": 1,
                "cons.conf.idx": -46.2,
                "cons.price.idx": 92.893,
                "contact": "cellular",
                "day_of_week": "mon",
                "default": "no",
                "duration": 971,
                "education": "university.degree",
                "emp.var.rate": -1.8,
                "euribor3m": 1.299,
                "housing": "yes",
                "job": "blue-collar",
                "loan": "yes",
                "marital": "married",
                "month": "may",
                "nr.employed": 5099.1,
                "pdays": 999,
                "poutcome": "failure",
                "previous": 1
              },
              {
                "age": 87,
                "campaign": 1,
                "cons.conf.idx": -46.2,
                "cons.price.idx": 92.893,
                "contact": "cellular",
                "day_of_week": "mon",
                "default": "no",
                "duration": 471,
                "education": "university.degree",
                "emp.var.rate": -1.8,
                "euribor3m": 1.299,
                "housing": "yes",
                "job": "blue-collar",
                "loan": "yes",
                "marital": "married",
                "month": "may",
                "nr.employed": 5099.1,
                "pdays": 999,
                "poutcome": "failure",
                "previous": 1
              },
          ]
        }
    # Convert to JSON string
    input_data = json.dumps(data)

Then we create a http post request with the scoring URI of the rest endpoint, the authentication key and the input data in json.

    # Set the content type
    headers = {'Content-Type': 'application/json'}
    # If authentication is enabled, set the authorization header
    headers['Authorization'] = f'Bearer {key}'

    # Make the request and display the response
    resp = requests.post(scoring_uri, input_data, headers=headers)
    print(resp.json())

Then we make the prediction and we got 2 results (No and No).

![Prediction](/img/endpoint_script.png)

### Endpoint Performance

Everything is working fine, we can now have performance check of our endpoint with Apache Benchmark.

![Apache Benchmark](/img/benchmark2.png)

10 requests were send, 0 failed and the mean time per request was 205ms.

Azure have a 60sec timeout so we are way below the threshold.

This wrapped up the process of training with AutoML and deploy it in order to make prediction from a rest endpoint.


### Pipeline

We now want to make a Pipeline object to encapsulate the AutoML training we did manualy earlier.

The pipeline will only have one step named autml_module:

    automl_step = AutoMLStep(
    name='automl_module',
    automl_config=automl_config,
    outputs=[metrics_data, model_data],
    allow_reuse=True)

In this step we call automl_config. This configuration is similar to the one we used earlier when we lauched the autoML run:

    automl_settings = {
        "experiment_timeout_minutes": 20,
        "max_concurrent_iterations": 5,
        "primary_metric" : 'AUC_weighted'
    }
    automl_config = AutoMLConfig(compute_target=compute_target,
                                 task = "classification",
                                 training_data=dataset,
                                 label_column_name="y",   
                                 path = project_folder,
                                 enable_early_stopping= True,
                                 featurization= 'auto',
                                 debug_log = "automl_errors.log",
                                 **automl_settings
                                )

Here we specify the task (Classification), the dataset (the one we registered previously), the target column ('y') and the primary metric we want to optimize ('AUC_weighted').

Then we submit the experiment in order to create the pipeline in Azure ML and lauch it:
    
    pipeline_run = experiment.submit(pipeline)

![Pipeline has been created](/img/Pipeline.png)

If we look inside the run, we can see the Dataset Object and the automl_module.

![Pipeline-Dataset + autoML](/img/Pipeline-Dataset.png)


After the pipeline run is completed, we can deploy the pipeline as an endpoint in order to launch it from outside Azure ML.

![Pipeline endpoints](/img/Pipeline-endpoints.png)

And the rest endpoint can be found here:

![Pipeline Published Overview](/img/PublishedPipelineOverview.png)

Now we can create a pipeline run from the rest endpoint with a http post request:

    import requests

    rest_endpoint = published_pipeline.endpoint
    response = requests.post(rest_endpoint, 
                             headers=auth_header, 
                             json={"ExperimentName": "pipeline-rest-endpoint"}
                            )

And have a look at the details with:

    RunDetails(published_pipeline_run).show()
    
![RunDetails](/img/RunDetails.png)


If we go back to Azure ML, we can see that we have a new run from the rest endpoint.

![Pipeline rest endpoint](/img/PipelinesRuns2.png)

That conclude this project, everything is pretty straightforward and the interaction between the client and the SDK is smooth.


## Screen Recording

https://youtu.be/Vt-t9fgTnKw


## How to improve



