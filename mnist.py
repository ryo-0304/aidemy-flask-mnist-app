import os

import numpy as np
from flask import Flask, flash, redirect, render_template, request
from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.preprocessing import image
from werkzeug.utils import secure_filename

classes = ["0", "1", "2", "3", "4", "5", "6", "7", "8", "9"]
image_size = 28

basedir = os.path.abspath(os.path.dirname(__file__))
UPLOAD_FOLDER = f"{basedir}/upload_files"
ALLOWED_EXTENSIONS = set(["png", "jpg", "jpeg", "gif"])

app = Flask(__name__)
model = load_model(f"{basedir}/models/model.keras")  # 学習済みモデルをロード

# app.secret_key = "your_secret_key_here"
# submitボタンを押した際にエラーが出た場合上の行のコメントアウトを削除し、your_secret_key_hereに任意の文字列（例:aidemy)を指定し、再度アプリケーションを実行してください。


def allowed_file(filename):
    return "." in filename and filename.rsplit(".", 1)[1].lower() in ALLOWED_EXTENSIONS


@app.route("/", methods=["GET", "POST"])
def upload_file():
    if not os.path.exists(UPLOAD_FOLDER):
        print("フォルダがありませんでした")
        os.makedirs(UPLOAD_FOLDER)

    if request.method == "POST":
        if "file" not in request.files:
            flash("ファイルがありません")
            return redirect(request.url)

        file = request.files["file"]
        if file.filename == "":
            flash("ファイルがありません")
            return redirect(request.url)

        if file and allowed_file(file.filename):
            filename = secure_filename(file.filename)
            file.save(os.path.join(UPLOAD_FOLDER, filename))
            filepath = os.path.join(UPLOAD_FOLDER, filename)

            # 受け取った画像を読み込み、np形式に変換
            img = image.load_img(
                filepath, color_mode="grayscale", target_size=(image_size, image_size)
            )
            img = image.img_to_array(img)
            data = np.array([img])

            # 変換したデータをモデルに渡して予測する
            result = model.predict(data)[0]
            predicted = result.argmax()
            pred_answer = "これは " + classes[predicted] + " です"

            return render_template("index.html", answer=pred_answer)

    return render_template("index.html", answer="")


if __name__ == "__main__":
    port = int(os.environ.get("PORT", 8080))
    app.run(host="0.0.0.0", port=port)
