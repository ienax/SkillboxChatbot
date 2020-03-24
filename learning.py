from tqdm import tqdm
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer
from sklearn.model_selection import train_test_split
from sklearn.linear_model import SGDClassifier
from sklearn.externals import joblib

with open('dialogues.txt', encoding='utf-8') as dialogues_file:
    content = dialogues_file.read()
    
dialogues = content.split('\n\n')
chit_chat_dataset = [] #[[qusetion, answer], ...]

for dialogue in dialogues:
    replicas = dialogue.split('\n')
    replicas = [replica[2:].strip().lower() for replica in replicas]
    replicas = [replica for replica in replicas if replica]
    for i in range(len(replicas) - 1):
        chit_chat_dataset.append((replicas[i], replicas[i+1]))
        
chit_chat_dataset = list(set(chit_chat_dataset))

print(chit_chat_dataset[:10])

print(len(chit_chat_dataset))

for dialogue in dialogues:
    for i in tqdm(range(int(len(chit_chat_dataset)))):
        X_text = [x for x, y in chit_chat_dataset]
        y = [y for x, y in chit_chat_dataset]
        
        pass
        
vectorizer = CountVectorizer(lowercase=True, ngram_range=(3,3), analyzer='char_wb')
X = vectorizer.fit_transform(X_text)  # вектора примеров

tfidf_transformer = TfidfTransformer()
x_tf = tfidf_transformer.fit_transform(X)
X = x_tf

scores = []
 
for _ in range(10000):
    for i in tqdm_notebook(range(int(10000)):
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3)
        clf = SGDClassifier(loss='hinge', penalty='l2',alpha=0.35e-3, max_iter=1000, random_state=3)
        clf.fit(X_train, y_train)
        score = clf.score(X_test, y_test)
        scores.append(score)
                           
        pass
 
result = sum(scores) / 10000
print(result)

joblib.dump(clf, 'my_model.pkl', compress=9)
