# fdb fra-oc-p2 28-09-19

import sys
import os
import pyspark
from pyspark.sql import SparkSession
from pyspark import SparkContext
from pyspark.sql import Row
import re
import numpy as np
from pyspark.mllib.classification import SVMWithSGD, SVMModel
from pyspark.mllib.regression import LabeledPoint
import datetime
from pyspark.sql.functions import lit

sc = SparkContext(appName="projet2_svm_whith_sgd")
spark = SparkSession.builder.getOrCreate()


# Script will run on cluster (True) or local (False)?
ClusterRun = True

# import results to csv and load it in S3
import csv
import boto3
s3 = boto3.client('s3')
s3_path = "s3://fra-oc-p2/"
local_path = "./"

if ClusterRun:
    path_doc = s3_path
else:
    path_doc = local_path
    import matplotlib.pyplot as plt

# *** Config :
# ** on laptop
# spark-defaults.conf -> spark.driver.memory 6g  (default: 512mo)
# ** on cluster:
# sudo yum update
# sudo sed -i -e '$a\export PYSPARK_PYTHON=/usr/bin/python3' /etc/spark/conf/spark-env.sh
# python3 -m pip install --user boto3

# *** run script as 
# spark-submit script.py path/to/json/ xxx yyy
# xxx => class1
# yyy => "vs" for test 1vs1 on each classes
# yyy => "All" for test 1vsAll

classes = [
    'beagle',
    'pomeranian',
    'great_pyrenees',
    'keeshond',
    'german_shorthaired',
    'Abyssinian',
    'basset_hound',
    'wheaten_terrier',
    'staffordshire_bull_terrier',
    'Bombay',
    'english_cocker_spaniel',
    'Sphynx',
    'saint_bernard',
    'japanese_chin',
    'Birman',
    'pug',
    'Persian',
    'newfoundland',
    'chihuahua',
    'shiba_inu',
    'samoyed',
    'american_pit_bull_terrier',
    'scottish_terrier',
    'Egyptian_Mau',
    'yorkshire_terrier',
    'Bengal',
    'British_Shorthair',
    'havanese',
    'leonberger',
    'Maine_Coon',
    'miniature_pinscher',
    'american_bulldog',
    'Ragdoll',
    'english_setter',
    'Russian_Blue',
    'boxer',
    'Siamese'
]


def convert_rdd_raw(row):
    # row[0] : full path to file
    classname = get_classname_from_filename(row[0])
    # row[1] : features as string value
    values = row[1].strip('[]').split(',')
    values = [float(x) for x in values]
    return pyspark.Row(label=classname, features=values)  


def get_classname_from_filename(filename):
    classname = re.sub(r'[0-9]', '', filename.split('/')[-1])
    classname = classname[:-9].strip('_')
    return classname

def train_model_and_test(df_data, class1, class2, split_list, seed_list, iteration_list):
    # global tab_csv for csv file
    global tab_csv
    model_number = 0
    best_score = 0.0
    graphic_x = []
    graphic_y = []

    # prepare train and test data for class1 and class2 (or All)
    print("\n***************************************************")
    print("Preparing for \"{}\" vs \"{}\"...\n".format(class1, class2))
    df_class1 = df_data.filter(df_data.label == class1).withColumn('label_idx', lit(float(0.0)))
    print("  class1:\"{}\", nb rows={}".format(class1, df_class1.count()))
    if class2 == 'All':
        df_class2 = df_data.filter(df_data.label != class1).withColumn('label_idx', lit(float(1.0)))
    else:
        df_class2 = df_data.filter(df_data.label == class2).withColumn('label_idx', lit(float(1.0)))
    print("  class2:\"{}\", nb rows={}".format(class2, df_class2.count()))
    print("\n... ready ! ")

    for cur_split in split_list:
        for cur_seed in seed_list:
            
            # Split the two dataframes according to split_values
            df_class1_split = df_class1.randomSplit([cur_split,1-cur_split], cur_seed)
            df_class2_split = df_class2.randomSplit([cur_split,1-cur_split], cur_seed)

            # Create two Dataframes (training and test) using union
            df_train = df_class1_split[0].union(df_class2_split[0])
            df_test = df_class1_split[1].union(df_class2_split[1])

            # count rows of each
            nb_rows_df_train = df_train.count() 
            nb_rows_df_test = df_test.count()

            # map rdd to labeledPOint (required with SVMWithSGD)
            rdd_train_idx = df_train.rdd.map(lambda line: LabeledPoint(line.label_idx, line.features))
            rdd_test_idx = df_test.rdd.map(lambda line: LabeledPoint(line.label_idx, line.features))

            print("\n*** Split= {:.2f} / {:.2f} ({} rows in train dataset, {} rows in test dataset)".format(cur_split*100, (1-cur_split)*100, nb_rows_df_train, nb_rows_df_test))

            for i in iteration_list:
                model_number += 1

                # train current model
                print("    ... Evaluating model #{} (split:{:.2f} %, iteration:{})".format(model_number, cur_split*100, i))
                cur_model = SVMWithSGD.train(rdd_train_idx, iterations=i)

                # test curent model
                rdd_test_predict = rdd_test_idx.map(lambda lblp: (lblp.label, cur_model.predict(lblp.features)))
                nb_correct_predictions = rdd_test_predict.filter(lambda line: line[0] == line[1]).count()

                # get current score and save it in csv file
                cur_score = float(nb_correct_predictions) / float(nb_rows_df_test)
                print("        ---> correct prediction: {:.2f} % ({} corrects / {} test images)".format(cur_score*100, nb_correct_predictions, nb_rows_df_test))
                tab_csv.append([class1, class2, cur_split, i, model_number, cur_score*100, nb_rows_df_train, nb_rows_df_test, nb_correct_predictions])
                
                # data for garphic perf/data-train
                graphic_x.append(cur_split*100)
                graphic_y.append(cur_score*100)

                # take the best 
                if cur_score >= best_score:
                    best_score = cur_score
                    best_model = model_number
                    best_model_type = cur_model
                    best_split = cur_split
                    best_iteration = i
                
    print("\n\nBest prediction: Model #{} whith {:.2f} % of success !".format(best_model, best_score*100))
    print("Model params: split: {:.2f}%, iterations: {}".format(best_split*100, best_iteration))  
    
    # Save model
    save_path = path_doc+"models/{}/pythonSVMWithSGDModel".format(str(class1+'_VS_'+class2))
    print("Save Model to: {} ...".format(save_path), end=" ", flush=True)
    best_model_type.save(sc, save_path)
    print(" Model saved\n")

    # graphic representation
    if not ClusterRun:
        plt.xlabel("% of train data on dataset")
        plt.xlabel("nb train features")
        plt.ylabel("accuracy - %")
        plt.title("Performances evolution according to the number \nof learning images (in % of dataset)")
        img_lab = str(class1+'_VS_'+class2)
        plt.plot(graphic_x, graphic_y, label=img_lab)
        plt.grid()
        plt.legend(loc='best')
        img_save_path = "Accuracy_{}.png".format(str(class1+'_VS_'+class2))
        plt.savefig(img_save_path, dpi=300)
        print("Image saved in : {}".format(img_save_path))
        print("Classification \"{}\" vs \"{}\" - END - ".format(class1, class2))
        # plt.show()
    else:
        print("Classification \"{}\" vs \"{}\" - END - ".format(class1, class2))

with open('logFile.txt', 'w') as f:
    # We redirect the 'sys.stdout' command towards the descriptor file
    if ClusterRun: sys.stdout = f
    
    # take params and control
    class1 = sys.argv[2] 

    if sys.argv[3] == "All":
        arg2 = "All" 
    elif sys.argv[3] in classes:
        arg2 = sys.argv[3]
    else:
        arg2 = ""

    if class1 in classes:
        print('\nparams OK')
    else:
        print('\nparams ERROR')
        sys.exit()

    # Application variable
    seed_list = [17]
    model_number = 0
    split_list = [0.3, 0.5, 0.8] # for more tests :  np.arange(0.10, 0.91, 0.10) -> from 10% to 90% split whith 10% steps
    iteration_list = [100] # for more tests : [50,100,150]
    best_score = 0.0
    vs_classes = []
    # init chrono
    timestamp_init = datetime.datetime.now() 
    """print("\nClassification \"{}\" vs \"{}\" - START - {} \n".format(class1, class2, (timestamp_init)))"""

    # get data  
    print("Get data", end=" ", flush=True)
    rdd_raw_data = sc.wholeTextFiles(sys.argv[1:][0]+'*.json') # wholeTextFiles ! to get classname
    # sc.wholeTextFiles(sys.argv[1:][0]+'*.json', minPartitions=24) # optimisation sur EMR
    t1 = (datetime.datetime.now()-timestamp_init).total_seconds()
    print("- OK - ({:.2f} sec.)".format(t1))

    # convert to row
    print("Transform data", end=" ", flush=True)
    rdd_data = rdd_raw_data.map(convert_rdd_raw)
    t2 = (datetime.datetime.now()-timestamp_init).total_seconds()
    print("- OK - ({:.2f} sec.)".format(t2-t1))

    # convert to dataframe
    print("Generate dataFrame", end=" ", flush=True)
    df_data = spark.createDataFrame(rdd_data).persist() 
    t3 = (datetime.datetime.now()-timestamp_init).total_seconds()
    print("- OK - ({:.2f} sec.)".format(t3-t2))

    # Init the Versus Class list
    # if arg2  == "All":
    if arg2  != "":
        vs_classes.append(arg2)
    else:
        vs_classes = [x for x in classes]
        # Remove class1 value from the Versus Class list
        vs_classes.remove(class1)
    

    # print the vs list
    print("vs_list: ")
    print(vs_classes)

    # open csv file to write results
    csv_file = open('results.csv', 'w')
    csv_writer = csv.writer(csv_file, delimiter=';', quotechar='"', quoting=csv.QUOTE_MINIMAL)
    tab_csv = [] # this list will have each csv row
    csv_writer.writerow(["class_1", "class_2", "split", "iteration", "num_model", "predict%", "nb_train", "nb_test", "nb_correct"])
    
    # for each value in vs list, train and test model whith parameters given. Then save best model. A timer is set
    for class2 in vs_classes:
        tloop_start = (datetime.datetime.now()-timestamp_init).total_seconds()
        train_model_and_test(df_data, class1, class2, split_list, seed_list, iteration_list)
        tloop_end = (datetime.datetime.now()-timestamp_init).total_seconds()
        print("Took {:.2f} sec.".format(tloop_end-tloop_start))

    # write results to csv file
    for row in tab_csv:
        csv_writer.writerow(row)
    csv_file.close()

    if ClusterRun:
        with open('logFile.txt', 'rb') as data:
            s3.upload_fileobj(data, 'fra-oc-p2', 'results.txt')
        with open('results.csv', 'rb') as data_csv:
            s3.upload_fileobj(data_csv, 'fra-oc-p2', 'results.csv')

    print("\n******************* END ***************************")
    # print chrono
    timestamp_end = datetime.datetime.now() 
    print("Program took {:.2f} sec. to perform.".format((timestamp_end-timestamp_init).total_seconds()))
    # *** Access to webUI Spark ***
    print("\n... You can go to webUI ...\n")
    # Exit app (only in local mode)
    if not ClusterRun: input("Press ctrl+c to exit")
