# Простая сессия через Flask
from flask import session, redirect

@app.route('/login', methods=['POST'])
def login():
    role = request.form.get('role')
    password = request.form.get('password')
    
    if role == 'admin' and password == os.getenv('ADMIN_PASSWORD'):
        session['role'] = 'admin'
        return redirect('/admin')
    # ...