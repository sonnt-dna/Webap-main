import numpy as np
import pandas as pd

# import lightgbm as lgb
import xgboost as xgb
import matplotlib.pyplot as plt
import pickle
import bz2file as bz2

from sklearn.model_selection import train_test_split, StratifiedShuffleSplit
from sklearn.metrics import roc_auc_score, balanced_accuracy_score, f1_score, recall_score, precision_score
from sklearn.metrics import roc_curve, precision_recall_curve, auc

import streamlit as st
from Antuns.p1_import_csv import upload_csv
from Antuns.page_setting import page_intro

page_intro()
# processing pipeline
def remove_negative_val(df, col):
    return df.drop(index=df[df[col] < 0].index)
def rel_depth(df):
    dfx = []
    for well in df.WELL.unique():
        df_ = df[df.WELL==well].sort_values(by="DEPTH", ascending=True)
        dfx.append(df_.assign(rel_depth=df_.DEPTH / df_.DEPTH.values[0]))
    return pd.concat(dfx).reset_index(drop=True)

def tweak_data(df):
    return (
        df.assign(
                FRACTURE_ZONE=df.FRACTURE_ZONE.replace({-9999: 0, np.nan: 0}).astype('int8'),
                GR=df.GR.replace({-9999.:0.}).astype('float32'),
                DCALI_FINAL=df.DCALI_FINAL.replace({-9999.:0.}).astype('float32'),
                LLD=df.LLD.replace({-9999.:0.}).astype('float32'),
                LLS=df.LLS.replace({-9999.:0.}).astype('float32'),
                NPHI=df.NPHI.replace({-9999.:0.}).astype('float32'),
                RHOB=df.RHOB.replace({-9999.:0.}).astype('float32'),
                DTC=df.DTC.replace({-9999.:0.}).astype('float32'),
                DTS=df.DTS.replace({-9999.:0.}).astype('float32'),
                DEPTH=df.DEPTH.astype('float32')
                )
                .pipe(remove_negative_val, "RHOB")
                .pipe(remove_negative_val, "DTC")
                .pipe(remove_negative_val, "DTS")
                .pipe(remove_negative_val, "GR")
                .pipe(remove_negative_val, "LLD")
            ).pipe(rel_depth)

#Main--------------------------------------------------------------------------------------
col_1, col_2, col_3 = st.columns([5,3,5])
with col_2:
    # st.image("https://i.ibb.co/Yd42K98/LogoVPI.png", width=250)
    st.header("Training Dashboard!")

#Loading data from browser:----------------------------------------
# st.subheader("1. Load CSV input file:")
df = upload_csv()
if df is not None:
    st.caption("Data Preparation")
    # Processing data
    df = tweak_data(df)
    st.info("Tweak Data")
    i1, i2 = st.columns(2)
    for i, v in enumerate(["FRACTURE_ZONE", "GR", "DCAL", "LLD", "LLS", "NPHI", "RHOB", "DTC", "DTS", "DEPTH"]):
        if i%2==0:
            with i1:
                st.success(f"{v}: Replaced nan values by 0")
        if i%2==1:
            with i2:
                st.success(f"{v}: Replaced nan values by 0")
    st.info(" Negative values removal in RHOB, DTC, DTS, GR, LLD: Done!")
    st.write("---")
    
    #--------------------------------------------------------------------------------------
    # define training/testing data
    feature_names = [col for col in df.columns if col not in ["WELL", 
                                                            "DEPTH", 
                                                            "Fracture Intensity",
                                                            "FRACTURE_ZONE",
                                                            ]]
    label_name = "FRACTURE_ZONE"
    st.caption("Features Selection")
    st.info(f"Label names: {label_name}")
    st.info(f"Feature names: {feature_names}")
    st.write("---")
    #--------------------------------------------------------------------------------------
    st.caption("Split Data")
    
    ## split data
    ### some data for test model after deploy
    ss = StratifiedShuffleSplit(n_splits=1, test_size=0.2, random_state=42)

    for _train_inx, _test_inx in ss.split(df, df["WELL"]):
        train_df, test_df = df.loc[_train_inx, :], df.loc[_test_inx, :]

    X_train, X_test, y_train, y_test = train_test_split(
                                                        train_df[feature_names],
                                                        train_df[label_name],
                                                        stratify=train_df[label_name],
                                                        train_size=0.9,
                                                        random_state=42,
                                                        )

    ### create lgb dataset
    train_set = xgb.DMatrix(X_train,
                            label=y_train,
                            # feature_name=feature_names,
                            )
    valid_set = xgb.DMatrix(X_test,
                            label=y_test,
                            # reference=train_set,
                            # feature_name=feature_names,
                            )
    
    st.info(f"Size of FULL Dataset: {len(df)}")
    # st.info(f"Size of TRAINING set: {train_set.construct().num_data()}")
    # st.info(f"Size of VALIDATION set: {valid_set.construct().num_data()}")
    st.info(f"Size of TESTING set: {len(test_df)}")
    st.write("---")
#Traning model--------------------------------------------------------------------------------------
    if st.button("Start Train"):
        # Modeling
        ## custom metric
        st.caption("Training")
        from sklearn.metrics import recall_score, precision_score, accuracy_score, f1_score, roc_auc_score, precision_recall_curve
        # model = lgb.train(
        model = xgb.train(
                        params={"booster": "gptree",
                                    "objective": "binary:logistic",
                                    # "metric": ["rmse","recall"],
                                    # "is_unbalance": True,
                                    },
                        Dtrain= train_set,
                        # xgb.DMatrix(data=X_train, label=y_train
                                            # feature_name=feature_names
                                            # ),
                                            num_boost_round=2000,
                        evals = valid_set,
                        # xgb.DMatrix(data=X_test, label=y_test,
                                            # feature_name=feature_names
                                            # ),
                                            early_stopping_rounds=5,
                                            verbose_eval=0,
                        )
        st.success("Finished Training!")
        st.success("Saved Model!")
        st.write("---")
        # model.save_model(filename='/Users/vpi103/Desktop/DnA/AI_Fractures/EDA_dataprep/Git-Fracture_WebApp/backup/model.json')
#Scores--------------------------------------------------------------------------------------
        ## using model to make prediction
        st.caption("Modeling Scores")
        
        threshold = 0.5
        test_preds = model.predict(xgb.DMatrix(test_df[feature_names]))
        train_preds = model.predict(xgb.DMatrix(X_train))
        valid_preds = model.predict(xgb.DMatrix(X_test))
        
        valid_recall = recall_score(y_test, valid_preds >= threshold, average = 'weighted')
        valid_precision = precision_score(y_test, valid_preds >= threshold, average = 'weighted')
        valid_acc = accuracy_score(y_test, valid_preds >= threshold)
        valid_f1 = f1_score(y_test, valid_preds >= threshold, average = 'weighted')
        valid_aoc = roc_auc_score(y_test, valid_preds >= threshold)

        train_recall = recall_score(y_train, train_preds >= threshold, average = 'weighted')
        train_precision = precision_score(y_train, train_preds >= threshold, average = 'weighted')
        train_acc = accuracy_score(y_train, train_preds >= threshold)
        train_aoc = roc_auc_score(y_train, train_preds >= threshold)
        train_f1 = f1_score(y_train, train_preds >= threshold, average = 'weighted')

        test_recall = recall_score(test_df[label_name], test_preds >= threshold, average = 'weighted')
        test_precision = precision_score(test_df[label_name], test_preds >= threshold, average = 'weighted')
        test_acc = accuracy_score(test_df[label_name], test_preds >= threshold)
        test_aoc = roc_auc_score(test_df[label_name], test_preds >= threshold)
        test_f1 = f1_score(test_df[label_name], test_preds >= threshold, average = 'weighted')
        
        sc1, sc2, sc3 = st.columns(3)
        with sc1:
            st.info(f"Training score (RECALL): {train_recall}")
            st.info(f"Training score (PRECISION): {train_precision}")
            st.info(f"Training score (ACC): {train_acc}")
            st.info(f"Training score (F1): {train_f1}")
            st.info(f"Training score (AOC): {train_aoc}")
            
        with sc2:
            st.info(f"Validation score (RECALL): {valid_recall}")
            st.info(f"Validation score (PRECISION): {valid_precision}")
            st.info(f"Validation score (ACC): {valid_acc}")
            st.info(f"Validation score (F1): {valid_f1}")
            st.info(f"Validation score (AOC): {valid_aoc}")
            
        with sc3:
            st.info(f"Testing score (RECALL): {test_recall}")
            st.info(f"Testing score (PRECISION): {test_precision}")
            st.info(f"Testing score (ACC): {test_acc}")
            st.info(f"Testing score (F1): {test_f1}")
            st.info(f"Testing score (AOC): {test_aoc}")
        st.write("---")
#Measure Scores--------------------------------------------------------------------------------------
        st.caption("Scores plotting charts")
        
        ## roc valid
        fpr_valid, tpr_valid, threshold_valid = roc_curve(y_test, valid_preds)
        roc_auc_valid = auc(fpr_valid, tpr_valid)
        ## precision recall valid
        pr_valid, rc_valid, threshold_valid= precision_recall_curve(y_test, valid_preds)
        ## roc training
        tfpr_train, ttpr_train, tthreshold_train = roc_curve(y_train, train_preds)
        troc_auc_train = auc(tfpr_train, ttpr_train)
        ## precision recall training
        tpr_train, trc_train, tthreshold_train = precision_recall_curve(y_train, train_preds)
        ## roc test
        tfpr_test, ttpr_test, tthreshold = roc_curve(test_df[label_name], test_preds)
        troc_auc_test = auc(tfpr_test, ttpr_test)
        ## precision recall testing
        tpr_test, trc_test, tthreshold_test = precision_recall_curve(test_df[label_name], test_preds)
    
#Plot Scores--------------------------------------------------------------------------------------
        fig, ax = plt.subplots(figsize=(40,40))
        ax1 = plt.subplot2grid((7,7), (0,0), rowspan=1, colspan = 1)
        ax2 = plt.subplot2grid((7,7), (0,1), rowspan=1, colspan = 1)
        ax3 = plt.subplot2grid((7,7), (0,2), rowspan=1, colspan = 1)
        ax4 = plt.subplot2grid((7,7), (1,0), rowspan=1, colspan = 1)
        ax5 = plt.subplot2grid((7,7), (1,1), rowspan=1, colspan = 1)
        ax6 = plt.subplot2grid((7,7), (1,2), rowspan=1, colspan = 1)
        
        def set_ax(ax, 
                x, y, color, label, legend, 
                line:bool=False,
                title:str=None,
                x_label:str=None,
                y_label:str=None,
                ):
            ax.plot(x, y, color, label=label)
            ax.set_title(title)
            ax.legend(loc = legend)
            if line == True:
                ax.plot([0, 1], [0, 1],'r--')
            ax.set_xlim([0, 1])
            ax.set_ylim([0, 1])
            ax.set_ylabel(y_label)
            ax.set_xlabel(x_label)
        p1, p2, p3 = st.columns([1,14,1])
        with p2:
            ## roc valid
            set_ax(ax2, fpr_valid, tpr_valid, 'b',
                label = 'AUC = %0.2f' % roc_auc_valid,
                legend='lower right',
                title='Receiver Operating Characteristic - Validation',
                line=True,
                x_label='False Positive Rate',
                y_label='True Positive Rate',
                )
            
            ## precision recall valid
            set_ax(ax5, pr_valid, rc_valid, 'orange',
                label = 'PR Curve',
                legend='lower right',
                title='Precision Recall Curve - Validation',
                line=True,
                x_label='Recall',
                y_label='Precision',
                )
        
            ## roc training
            set_ax(ax1, tfpr_train, ttpr_train, 'b',
                label = 'AUC = %0.2f' % troc_auc_train,
                legend='lower right',
                title='Receiver Operating Characteristic - Training',
                line=True,
                x_label='False Positive Rate',
                y_label='True Positive Rate',
                )
            
            ## precision recall training
            set_ax(ax4, tpr_train, trc_train, 'orange',
                label = 'PR Curve',
                legend='lower right',
                title='Precision Recall Curve - Training',
                line=True,
                x_label='Recall',
                y_label='Precision',
                )

            ## roc test
            set_ax(ax3, tfpr_test, ttpr_test, 'b',
                label = 'AUC = %0.2f' % troc_auc_test,
                legend='lower right',
                title='Receiver Operating Characteristic - Blind test',
                line=True,
                x_label='False Positive Rate',
                y_label='True Positive Rate',
                )
            
            ## precision recall testing
            set_ax(ax6, tpr_test, trc_test, 'orange',
                label = 'PR Curve',
                legend='lower right',
                title='Precision Recall Curve - Blind test',
                line=True,
                x_label='Recall',
                y_label='Precision',
                )
            
            st.pyplot(fig)
            st.snow()