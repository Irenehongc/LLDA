from sklearn import cross_validation
from sklearn import datasets
from sklearn.preprocessing import MultiLabelBinarizer
from sklearn.metrics import f1_score

from get_vector import Dataset
from llda import LLDAClassifier


def get_data():
    f = open("movie_info", 'r', encoding="utf-8")
    comment = []
    comment_rating = []
    for lines in f.readlines():
        comment_data = []
        lines = lines.strip()
        lines = lines.split('\t')
        comment_data.append(lines[-2])
        comment_rating.append(lines[-1])
        comment.append(comment_data)
    return comment, comment_rating


def main():
    # comments, comment_rating = get_data()
    # iris = datasets.load_iris()
    # iris.data.shape, iris.target.shape
    # print(iris.data)
    # print(iris.target)
    # X_train, X_test, y_train, y_test = cross_validation.train_test_split(comments, comment_rating, test_size=0.4,
    #                                           random_state=0)

    dataset = Dataset()
    dataset.get_dataset()
    x_train = dataset.train_data
    x_test = dataset.test_data
    # print(x_test)
    y_train = dataset.train_target
    y_test = dataset.test_target

    mlb = MultiLabelBinarizer()
    y_train = [[each] for each in y_train]
    y_train = mlb.fit_transform(y_train)
    llda = LLDAClassifier(alpha=0.5 / y_train.shape[1])
    print("4")
    # x_train = [[(),(), ... ()],[],[]]   20个turple
    llda.fit(x_train, y_train)
    print("5")
    result = mlb.fit_transform(llda.predict(x_test, assignment=True))
    print("6")
    y_test = mlb.fit_transform([[each] for each in y_test])

    score_macro = f1_score(y_test, result, average="macro")
    score_micro = f1_score(y_test, result, average="micro")
    print("F1_macro:{0}, F1_micro:{1}".format(score_macro, score_micro))


if __name__ == '__main__':
    main()
