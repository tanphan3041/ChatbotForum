import nltk
nltk.download('popular')
from nltk.stem import WordNetLemmatizer
lemmatizer = WordNetLemmatizer()
import pickle
import numpy as np
from keras.models import load_model
model = load_model('model.h5')
import json
import random
intents = json.loads(open('data.json').read())
words = pickle.load(open('texts.pkl','rb'))
classes = pickle.load(open('labels.pkl','rb'))

import underthesea
from flask import Flask, flash, render_template, request, redirect, url_for
from flask_login import UserMixin, LoginManager, logout_user, login_user, current_user, login_required
from flask_sqlalchemy import SQLAlchemy
from werkzeug.security import generate_password_hash, check_password_hash
from waitress import serve



def clean_up_sentence(sentence):
    sentence_words = underthesea.word_tokenize(sentence)
    sentence_words = [lemmatizer.lemmatize(word.lower()) for word in sentence_words]
    return sentence_words

def bow(sentence, words, show_details=True):
    sentence_words = clean_up_sentence(sentence)
    bag = [0]*len(words)  
    for s in sentence_words:
        for i,w in enumerate(words):
            if w == s: 
                bag[i] = 1
                if show_details:
                    print ("found in bag: %s" % w)
    return(np.array(bag))

def predict_class(sentence, model):
    # filter out predictions below a threshold
    p = bow(sentence, words,show_details=False)
    res = model.predict(np.array([p]))[0]
    ERROR_THRESHOLD = 0.25
    results = [[i,r] for i,r in enumerate(res) if r>ERROR_THRESHOLD]
    # sort by strength of probability
    results.sort(key=lambda x: x[1], reverse=True)
    return_list = []
    for r in results:
        return_list.append({"intent": classes[r[0]], "probability": str(r[1])})
    return return_list

def getResponse(ints, intents_json):
    tag = ints[0]['intent']
    list_of_intents = intents_json['intents']
    for i in list_of_intents:
        if(i['tag']== tag):
            result = random.choice(i['responses'])
            break
    return result

def chatbot_response(msg):
    ints = predict_class(msg, model)
    res = getResponse(ints, intents)
    return res



### Flask app config
app = Flask(__name__)
app.app_context().push()

app.config['SQLALCHEMY_DATABASE_URI'] = 'sqlite:///database.sqlite3'
app.config['SEND_FILE_MAX_AGE_DEFAULT'] = 0
app.secret_key = 'Secret'

app.static_folder = 'static'



### Database config
login_manager = LoginManager(app)
db = SQLAlchemy(app)

class User(UserMixin, db.Model):
    id = db.Column(db.Integer, primary_key=True)
    email = db.Column(db.String(100), unique=True, nullable=False)
    password = db.Column(db.String(255), nullable=False)
    comments = db.relationship('Comment', backref='user')
    posts = db.relationship('Post', backref='user')

    def __repr__(self):
        return f'<User "{self.email}">'
    
    def set_password(self, password):
        self.password = generate_password_hash(password)

    def check_password(self, password):
        return check_password_hash(self.password, password)

class Post(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    user_id = db.Column(db.Integer, db.ForeignKey('user.id'))
    title = db.Column(db.String(100))
    body = db.Column(db.Text)
    created_at = db.Column(db.DateTime(timezone=True), server_default=db.func.now())
    comments = db.relationship('Comment', backref='post')

    def __repr__(self):
        return f'<Post "{self.title}">'

class Comment(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    user_id = db.Column(db.Integer, db.ForeignKey('user.id'))
    post_id = db.Column(db.Integer, db.ForeignKey('post.id'))
    body = db.Column(db.Text)
    created_at = db.Column(db.DateTime(timezone=True), server_default=db.func.now())

    def __repr__(self):
        return f'<Comment "{self.body[:20]}...">'

db.create_all()



### Website route
@app.route('/')
def main_page():
    posts = Post.query.all()
    if request.args.get('filter') == 'most-recent':
        posts = Post.query.order_by(Post.created_at.desc()).all()
    return render_template("register.html", posts = posts)

@app.route('/users', methods=['POST'])
def create_user():
    user = User.query.filter_by(email = request.form['email']).first()
    if user:
        flash('Người dùng đã tồn tại', 'danger')
        return redirect(url_for('main_page'))
    else:
        if request.method == 'POST':
            user = User(email = request.form['email']) 
            user.set_password(request.form['password'])
            flash('Đăng ký thành công', 'success')
            db.session.add(user)
            db.session.commit()
            login_user(user)
            return redirect(url_for('main_page'))

@app.route('/login')
def login():
    return render_template('login.html')

@app.route('/login', methods=['POST'])
def login_post():
    email = request.form.get('email')
    password = request.form.get('password')
    remember = True if request.form.get('remember') else False

    user = User.query.filter_by(email=email).first()
    if not user or not user.check_password(password):
        flash('Sai tài khoản hoặc mật khẩu.', 'danger')
        return redirect(url_for('login'))

    login_user(user, remember=remember)
    return redirect(url_for('main_page'))

@app.route('/logout')
def logout():
    logout_user()
    return render_template('register.html')

@app.route('/posts', methods=['POST'])
def create_post():
    if request.method == 'POST':
        post = Post(body = request.form['body'], 
                    user_id = current_user.id,
                    title = request.form['title']
        )
        db.session.add(post)
        db.session.commit()
        return redirect(url_for('main_page'))

@app.route('/posts/<id>')
def view_post(id):
    post = Post.query.get(id)
    comments = Comment.query.filter_by(post_id = post.id).all()
    return render_template('post.html', post = post, comments = comments)

@app.route('/posts/<id>/comments', methods=['POST'])
def create_comment(id):
    comment = Comment(user_id = current_user.id, 
                      post_id = id, 
                      body = request.form['body'])
    db.session.add(comment)
    db.session.commit()
    flash('Cảm ơn vì bình luận của bạn', 'success')
    return redirect(url_for('view_post', id = id))

@app.route('/posts/<id>', methods=['POST'])
def edit_post(id):
    post = Post.query.get(id)
    post.title = request.form['title']
    post.body = request.form['body']
    db.session.commit()
    return redirect(url_for('main_page'))

@app.route('/posts/<id>/delete')
def delete_post(id):
    post = Post.query.get(id)
    db.session.delete(post)
    db.session.commit() 
    return redirect(url_for('main_page'))

@login_manager.user_loader
def load_user(user_id):
    return User.query.get(user_id)

@app.route('/search')
def search():
    return render_template('search.html')    

@app.route('/search', methods=['post'])
def search_post():
    search = request.form.get('search')
    q = db.session.query(Post).filter(Post.title.like(search)).all()
    return redirect(url_for("search", q = q))



### Chatbot route
@app.route("/chat")
def chat():
    return render_template("chatbot.html")

@app.route("/get")
def get_bot_response():
    userText = request.args.get('msg')
    return chatbot_response(userText)


if __name__ == "__main__":
#    serve(app, host="0.0.0.0", port=3000)
   app.run(debug=True)