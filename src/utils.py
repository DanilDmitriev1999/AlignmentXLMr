from sklearn.metrics import classification_report


def report(label, predict):
    flatten = lambda t: [item for sublist in t for item in sublist]
    label = [i for i in flatten(label)]
    predict = [i for i in flatten(predict)]

    print(classification_report(label, predict))