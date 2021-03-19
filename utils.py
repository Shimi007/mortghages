import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn import preprocessing
from sklearn.linear_model import LogisticRegression
import settings

HEADER = 0


def trim_quotes_from_header(data):
    new_headers = []

    for header in data.columns:  # data.columns is your list of headers
        header = header.strip('"')  # Remove the quotes off each header
        new_headers.append(header)  # Save the new strings without the quotes

    data.columns = new_headers  # Replace the old headers with the new list

def trim_comas_from_header(data):
    new_headers = []

    for header in data.columns:  # data.columns is your list of headers
        header = header.strip(',')  # Remove the quotes off each header
        new_headers.append(header)  # Save the new strings without the quotes

    data.columns = new_headers  # Replace the old headers with the new list


def plotting_data_for_analyse_distribution_of_cardinal(df, predict):
    # Morgages
    print(df["mortage_yn"].value_counts(normalize=True) * 100)
    df["mortage_yn"].value_counts(normalize=True).plot.bar(title='mortage_yn', color=['lavender', 'green'])
    plt.show()

    # Plotting genre
    df['gender'].value_counts(normalize=True).plot.bar(title='Gender', color=['lavender', 'green'])
    plt.show()

    # martial_status
    print(df["martial_status"].value_counts(normalize=True) * 100)
    df['martial_status'].value_counts(normalize=True).plot.bar(title='Martial Status',
                                                               color=['lavender', 'green', 'purple', 'gray', 'red'])
    plt.show()

    # Education
    df['education'].value_counts(normalize=True).plot.bar(title='Education',
                                                          color=['lavender', 'green', 'purple', 'gray', 'red', 'blue', 'orange', 'pink'])
    plt.show()

    # Employment
    df['employment'].value_counts(normalize=True).plot.bar(title='Employment',
                                                           color=['lavender', 'green', 'purple', 'gray', 'red'])
    plt.show()

    #  Potential customer martial_status
    predict['martial_status'].value_counts(normalize=True).plot.bar(title='Potential Customer Martial Status',
                                                               color=['lavender', 'green', 'purple', 'gray', 'red'])
    plt.show()

    # Potential customer Education
    predict['education'].value_counts(normalize=True).plot.bar(title='Potential Customer Education',
                                                          color=['lavender', 'green', 'purple', 'gray', 'red', 'blue', 'orange', 'pink'])
    plt.show()

    # Potential customer Employment
    predict['employment'].value_counts(normalize=True).plot.bar(title='Potential Customer Employment',
                                                                color=['lavender', 'green', 'purple', 'gray', 'red'])
    plt.show()

    # Gender & mortgage
    Gender = pd.crosstab(df["gender"], df["mortage_yn"])
    Gender.div(Gender.sum(1).astype(float), axis=0).plot(kind="bar", stacked=True, figsize=(4, 4),
                                                         color=['green', 'purple'])
    plt.xlabel("Gender")
    plt.ylabel("Percentage")
    plt.show()

    # Martial_status & mortgage
    martial_status = pd.crosstab(df["martial_status"], df["mortage_yn"])
    martial_status.div(martial_status.sum(1).astype(float), axis=0).plot(kind="bar", stacked=True, figsize=(4, 4),
                                                                         color=['lavender', 'green', 'purple', 'gray',
                                                                                'red'])
    plt.xlabel("Martial Status")
    plt.ylabel("Percentage")
    plt.show()

    # Education & mortgage
    education = pd.crosstab(df["education"], df["mortage_yn"])
    education.div(education.sum(1).astype(float), axis=0).plot(kind="bar", stacked=True, figsize=(4, 4),
                                                               color=['lavender', 'green', 'purple', 'gray',
                                                                      'red'])
    plt.xlabel("education")
    plt.ylabel("Percentage")
    plt.show()

    # Employment & mortgage
    employment = pd.crosstab(df["employment"], df["mortage_yn"])
    employment.div(employment.sum(1).astype(float), axis=0).plot(kind="bar", stacked=True, figsize=(4, 4),
                                                                 color=['lavender', 'green', 'purple', 'gray',
                                                                        'red'])
    plt.xlabel("employment")
    plt.ylabel("Percentage")
    plt.show()


def plotting_data_analyse_distribution_of_numeric(df):
    # cust_income
    df["cust_income"].plot.box(figsize=(16, 5), title='Customer income')
    plt.show()

    # current_balance_eur
    df["current_balance_eur"].plot.box(figsize=(16, 5), title='Current balance eur')
    plt.show()

def visualize_confusion_matrix(cnf_matrix):
    class_names = [0, 1]  # name  of classes
    fig, ax = plt.subplots()
    tick_marks = np.arange(len(class_names))
    plt.xticks(tick_marks, class_names)
    plt.yticks(tick_marks, class_names)
    # create heatmap
    sns.heatmap(pd.DataFrame(cnf_matrix), annot=True, cmap="YlGnBu", fmt='g')
    ax.xaxis.set_label_position("top")
    plt.tight_layout()
    plt.title('Confusion matrix', y=1.1)
    plt.ylabel('Actual label')
    plt.xlabel('Predicted label')
    plt.savefig(settings.save_confusion_matrix_plot)


def potential_customers(prediction_set, LR):
    # Setting Features and y
    X_test = np.asarray(
        prediction_set[['age', 'years_with_bank', 'martial_status', 'education', 'employment', 'gender', 'cust_income',
                        'current_address', 'current_job', 'current_balance_eur']])

    # Normalization of dataset
    X_test = preprocessing.StandardScaler().fit(X_test).transform(X_test)
    y_test = LR.predict(X_test)
    return y_test


def preprocessing_data_for_writing(prediction_set, y_test):
    prediction_set['mortage_yn'] = y_test
    print(prediction_set.to_csv(settings.save_mortgage_path))


# def plotting_data_after_training_potential_customers():
#     prediction_set = pd.read_csv(settings.save_mortgage_path, sep=',', engine='python')
#     trim_comas_from_header(prediction_set)
#
#     # Morgages
#     print(prediction_set["mortage_yn"].value_counts(normalize=True) * 100)
#     prediction_set["mortage_yn"].value_counts(normalize=True).plot.bar(title='Potential Customers Mortages', color=['lavender', 'green'])
#     plt.savefig('/Users/shimi/Desktop/data science/mortages/plots/test2.png')
