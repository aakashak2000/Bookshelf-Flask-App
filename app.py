import os
from werkzeug.utils import secure_filename
from flask import Flask, send_from_directory, Response, flash, session, request, redirect, url_for, render_template
from books import *
from flask_wtf import FlaskForm
from wtforms import SelectField, SubmitField
import pandas as pd


UPLOAD_FOLDER = 'static/'
ALLOWED_EXTENSIONS = {'jpg', 'png', 'jpeg'}

# HOUR_CHOICES = [('1', '8am'), ('2', '10am')]


app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
app.config['SECRET_KEY'] = 'MYSECRETKEY'

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

books = get_books()
book_choices = [(book, book) for book in books]
# print(book_choices)
libs = ['Crossword', 'Granth', 'BookPlaza']
lib_choices = [(lib, lib) for lib in libs]

def get_shelves(library):
    df = pd.read_csv('library_data/Master_data.csv',encoding = "ISO-8859-1" )
    shelves = list(set(df[df.Library == library].shelf.to_list()))
    return shelves


class BookForm(FlaskForm):
    book = SelectField('What Book are you looking for?', choices=book_choices)
    submit = SubmitField('Submit')

class LibForm(FlaskForm):
    lib = SelectField('Which book store do you work for?', choices=lib_choices)
    submit = SubmitField('Submit')

class ShelfForm(FlaskForm):
    shelf = SelectField('Which shelf are you interested in?', coerce=int)
    submit = SubmitField('Submit')



@app.route("/")
def index():
    return render_template('index.html')

# @app.route("/employee")
# def employee():
#     return render_template('choose_lib.html')

@app.route('/customer')
def customer():
    return render_template('customer_home.html')

@app.route("/books", methods = ['GET', 'POST'])
def view_books():
    book_sel = None
    form = BookForm()
    if form.validate_on_submit():
        session['book'] = form.book.data
        return redirect(url_for('book_details'))
    return render_template('choose_book.html', form = form, books_list = books, book_seleced = book_sel)

@app.route('/employee', methods = ['GET', 'POST'])
def view_libraries():
    lib_sel = None
    form = LibForm()
    if form.validate_on_submit():
        session['lib'] = form.lib.data
        libr = session['lib']
  
        return redirect(url_for('choose_shelf'))
    return render_template('choose_lib.html', form = form, library_list = libs, library_selected = lib_sel)
   
# print(session['lib'])
# shelves = get_shelves()
# shelf_choices = [(s, s) for s in shelves]
# print(shelf_choices)



@app.route('/choose_shelf', methods = ['GET', 'POST'])
def choose_shelf():
    shelf_sel = None
    libr = session['lib']
    shelves = get_shelves(libr)
    print(shelves)
    # shelves = ['abc', 'def']
    shelves_list = [(i, s) for i,s in enumerate(shelves)]
    print(shelves_list)
    form = ShelfForm()
    form.shelf.choices = shelves_list
    print(form.shelf.choices)
    if form.validate_on_submit():
        session['shelf'] = form.shelf.data
        return redirect(url_for('shelf_details'))
    return render_template('view_shelves.html', form = form, shelf_list = shelves, shelf_selected = shelf_sel)
   

@app.route('/shelf_details', methods = ['GET', 'POST'])
def shelf_details():
    return render_template('shelf_details.html')


@app.route('/book', methods = ['GET', 'POST'])
def book_details():
    print(session['book'])
    book_deets = get_book_pic(session['book'])
    filename = '/static/'+book_deets['filename']
    lib = book_deets['library']
    shelf = book_deets['shelf']
    details = get_book_details(session['book'])
    details = details.split('\n')
    print(filename)
    return render_template('book_details.html', fname = filename, lib = lib, shelf = shelf, details = details)



@app.route('/shelf_display', methods = ['GET', 'POST'])
def shelf_display():
    current_lib = session['lib']
    shelves = get_shelves(current_lib)
    current_shelf = shelves[session['shelf']]
    loc = get_loc(current_lib, current_shelf)
    
    infos = get_infos(current_lib, current_shelf, loc)
    
    print(infos)
    # image = increase_brightness(image, 40)
    # plot_img(image)
    
    filename = '/static/display_shelf.jpg'
    print(filename)
    return render_template('display_details.html', fname = filename, infos = infos, lib = current_lib, shelf = current_shelf)


@app.route('/lib', methods = ['GET', 'POST'])
def lib_details():
    return render_template('lib.html')

@app.route('/upload')
def upload_file():
   return render_template('update_shelf.html')
	
@app.route('/update', methods = ['GET', 'POST'])
def update_shelf():
    print(request.method)
    if request.method == 'POST':
        print('rq method: POST')
        # check if the post request has the file part
        if 'file' not in request.files:
            print('No file part')
            flash('No file part')
            return redirect(request.url)
        file = request.files['file']
        # if user does not select file, browser also
        # submit an empty part without filename
        if file.filename == '':
            flash('No selected file')
            return redirect(request.url)

        if file:
            print('if file True')
            filename = secure_filename(file.filename)
            print(filename)
            file.save(os.path.join(app.config['UPLOAD_FOLDER'], filename))
            img = cv2.imread(os.path.join(app.config['UPLOAD_FOLDER'], filename))
            new, points = draw_spine_lines(img)
            filename_new = 'modified.jpg'
            cv2.imwrite(os.path.join(app.config['UPLOAD_FOLDER'], filename_new), new)
            lib = session['lib']
            shelf = session['shelf']
            shelves = get_shelves(lib)
            shelf = shelves[shelf]
            df = pd.read_csv('library_data/Master_data.csv',encoding = "ISO-8859-1")
            books = (df[(df.Library == lib) & (df.shelf == shelf)].Book.to_list())
            books = list(dict.fromkeys(books))
            count = 0
            missing = []
            for book in books:
                print(book, lib, shelf)
                is_present, dets = find_presence_of_book(book, lib, shelf, img)
                if is_present==False:
                    count+=1
                    missing.append(books)
            print(count, missing) 
            filename_new = 'static/modified.jpg'
            return render_template('update_shelf.html', count = count, missing = missing, filename=filename_new, lib = lib, shelf=shelf)
    return render_template('update_shelf.html')


# @app.route('/submitAudio', methods=['GET', 'POST'])
# def upload_file():
#     if request.method == 'POST':
#         # check if the post request has the file part
#         if 'file' not in request.files:
#             flash('No file part')
#             return redirect(request.url)
#         file = request.files['file']
#         # if user does not select file, browser also
#         # submit an empty part without filename
#         if file.filename == '':
#             flash('No selected file')
#             return redirect(request.url)
#         if file:
#             filename = secure_filename(file.filename)
#             file.save(os.path.join(app.config['UPLOAD_FOLDER'], filename))
#             aud = AudioSegment.from_file(os.path.join(app.config['UPLOAD_FOLDER'], filename))
#             script = generate_script_from_audio(os.path.join(app.config['UPLOAD_FOLDER'], filename))
#             txtname = filename[:-4] + '.txt'            
#             f = open(os.path.join(app.config['UPLOAD_FOLDER'], txtname), 'w')
#             f.write(script)
#             f.close()       
#             return redirect(url_for('uploaded_file',
#                                     txtname=txtname))
#         return render_template('textFile.html',text=script,filename=filename)


if __name__ == '__main___':
    app.run(debug=True)
