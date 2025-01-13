// Copyright 2023 The MediaPipe Authors.

// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//      http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.
import {
    PoseLandmarker,
    FilesetResolver,
    DrawingUtils
} from "https://cdn.skypack.dev/@mediapipe/tasks-vision@0.10.0";

const demosSection = document.getElementById("demos");

let poseLandmarker = undefined;
let runningMode = "IMAGE";
let imageSegmenter;
// Before we can use PoseLandmarker class we must wait for it to finish
// loading. Machine Learning models can be large and take a moment to
// get everything needed to run.
const loadingInterval = setInterval(() => {
    document.getElementById("loadingMsg").innerHTML += ".";
    if (document.getElementById("loadingMsg").innerHTML.length > 25) {
        document.getElementById("loadingMsg").innerHTML = "モジュールの読み込みに数秒かかります...";
    }
}, 1000);

const createPoseLandmarker = async () => {
    const vision = await FilesetResolver.forVisionTasks(
        "https://cdn.jsdelivr.net/npm/@mediapipe/tasks-vision@0.10.0/wasm"
    );
    poseLandmarker = await PoseLandmarker.createFromOptions(vision, {
        baseOptions: {
            modelAssetPath: `https://storage.googleapis.com/mediapipe-models/pose_landmarker/pose_landmarker_lite/float16/1/pose_landmarker_lite.task`,
            delegate: "GPU"
        },
        runningMode: runningMode,
        numPoses: 2
    });
    demosSection.classList.remove("invisible");
    document.getElementById("loadingMsg").style.display = "none";
    clearInterval(loadingInterval);

    runPoseEstimation();
    // 骨格画像をダウンロードする仕組みを追加
    let download_canvas_before = document.getElementById("download_canvas_before");
    download_canvas_before.href = document.getElementById("canvas_before").toDataURL();
    download_canvas_before.download = "pose_before.png";
    // download_canvas_before.style.display = "";




};
createPoseLandmarker();

const positionNamesJP = [
    "鼻 (nose)",
    "左目-内側 (left eye - inner)",
    "左目 (left eye)",
    "左目-外側 (left eye - outer)",
    "右目-内側 (right eye - inner)",
    "右目 (right eye)",
    "右目-外側 (right eye - outer)",
    "左耳 (left ear)",
    "右耳 (right ear)",
    "口-左縁 (mouth - left)",
    "口-右縁 (mouth - right)",
    "左肩 (left shoulder)",
    "右肩 (right shoulder)",
    "左肘 (left elbow)",
    "右肘 (right elbow)",
    "左手首 (left wrist)",
    "右手首 (right wrist)",
    "左小指 (left pinky)",
    "右小指 (right pinky)",
    "左人差し指 (left index)",
    "右人差し指 (right index)",
    "左親指 (left thumb)",
    "右親指 (right thumb)",
    "左腰 (left hip)",
    "右腰 (right hip)",
    "左膝 (left knee)",
    "右膝 (right knee)",
    "左足首 (left ankle)",
    "右足首 (right ankle)",
    "左かかと (left heel)",
    "右かかと (right heel)",
    "左足先 (left foot index)",
    "右足先 (right foot index)"
];
/********************************************************************
// Demo 1: Grab a bunch of images from the page and detection them
// upon click.
********************************************************************/

// In this demo, we have put all our clickable images in divs with the
// CSS class 'detectionOnClick'. Lets get all the elements that have
// this class.

// すでにアップされている画像を選択して表示し、ポーズ推定を行う
let result_before = [];
const image_before = document.getElementById("image_before");
const canvas_overlay = document.getElementById("canvas_overlay");
const canvas_before = document.getElementById("canvas_before");
canvas_overlay.width = image_before.width;
canvas_overlay.height = image_before.height;
canvas_overlay.style.top = image_before.offsetTop;
canvas_overlay.style.left = 0;
canvas_before.width = image_before.width;
canvas_before.height = image_before.height;

// 画像が表示されたら、ポーズ推定を行う
const runPoseEstimation = () => {
    // image_before.onload = () => {
    console.log("image_before.onload");
    if (!poseLandmarker) {
        console.log("Wait for poseLandmarker to load before clicking!");
        return;
    }

    if (runningMode === "VIDEO") {
        runningMode = "IMAGE";
        poseLandmarker.setOptions({ runningMode: "IMAGE" });
    }
    // poseLandmarker.setOptions({ outputSegmentationMasks: true });



    poseLandmarker.detect(image_before, async (result) => {
            // サンプル画像をcanvasに描画する仕組み: https://www.g-u-k.jp/take_log/archives/826 drawImage()を使う
    const sample_image_src = "./sample.jpg";
    const sample_image = new Image();
    sample_image.src = sample_image_src;
    sample_image.onload = () => {
        const sample_img_canvas = document.getElementById("sample_img_canvas");
        sample_img_canvas.width = sample_image.width;
        sample_img_canvas.height = sample_image.height;
        const sample_img_canvasCtx = sample_img_canvas.getContext("2d");
        sample_img_canvasCtx.drawImage(sample_image, 0, 0);
        // sample_img_canvas.style.display = "";
        // この画像に骨格画像を重ねる

    
        const canvas_overlayCtx = canvas_overlay.getContext("2d");
        const canvas_beforeCtx = canvas_before.getContext("2d");
        const drawingUtils_overlay = new DrawingUtils(canvas_overlayCtx);
        const drawingUtils_before = new DrawingUtils(canvas_beforeCtx);
        const DrawingUtils_sample = new DrawingUtils(sample_img_canvasCtx);
        for (const landmark of result.landmarks) {
            drawingUtils_overlay.drawLandmarks(landmark, {
                radius: (data) => DrawingUtils.lerp(data.from?.z ?? 0, -0.15, 0.1, 5, 1),
                lineWidth: 2,
            });
            drawingUtils_before.drawLandmarks(landmark, {
                radius: (data) => DrawingUtils.lerp(data.from?.z ?? 0, -0.15, 0.1, 5, 1),
                lineWidth: 2,
            });
            DrawingUtils_sample.drawLandmarks(landmark, {
                radius: (data) => DrawingUtils.lerp(data.from?.z ?? 0, -0.15, 0.1, 5, 1),
                lineWidth: 2,
            });
            drawingUtils_overlay.drawConnectors(landmark, PoseLandmarker.POSE_CONNECTIONS,
                { lineWidth: 2, color: "white" });
            drawingUtils_before.drawConnectors(landmark, PoseLandmarker.POSE_CONNECTIONS,
                { lineWidth: 2, color: "white" });
            DrawingUtils_sample.drawConnectors(landmark, PoseLandmarker.POSE_CONNECTIONS,
                { lineWidth: 2, color: "white" });
        }



        result_before = result;
        console.log("result_before : ");
        console.log(result_before);
        document.getElementById("fileSelector").disabled = false;
        document.getElementById("loadingMsg2").innerText = "比較したい画像を選択してください▽";

        // sample_img_canvasをダウンロードする仕組みを追加
        let download_sample_img_canvas = document.getElementById("download_sample_img_canvas");
        download_sample_img_canvas.href = sample_img_canvas.toDataURL();
        download_sample_img_canvas.download = "pose_sample.png";
        download_sample_img_canvas.style.display = "";

    };
    })
};


// 手持ちの画像を選択して表示し、ポーズ推定を行う

const FileSelector = document.getElementById("fileSelector");
const image_after = document.getElementById("selectedImage");
const canvas_after = document.getElementById("canvas_after");
const canvas_after_overlay = document.getElementById("canvas_after_overlay");
const download_canvas = document.getElementById("download_canvas");

let result_after = [];
FileSelector.addEventListener("change", (event) => {
    const file = event.target.files[0];
    if (!file) {
        return;
    }
    const reader = new FileReader();
    reader.onload = (e) => {
        image_after.src = e.target.result;
    };
    reader.readAsDataURL(file);
    // 画像が表示されたら、ポーズ推定を行う
    image_after.onload = () => {
        canvas_after.style.display = "";
        canvas_after.width = image_after.width;
        canvas_after.height = image_after.height;
        canvas_after.style.top = image_after.width;
        canvas_after.style.left = "2em";
        canvas_after_overlay.style.display = "";
        canvas_after_overlay.width = image_after.width;
        canvas_after_overlay.height = image_after.height;
        canvas_after_overlay.style.top = image_after.width;
        canvas_after_overlay.style.left = "2em";
        const canvas_auxiliary = document.getElementById("canvas_auxiliary");
        canvas_auxiliary.style.display = "";
        canvas_auxiliary.width = image_after.width;
        canvas_auxiliary.height = image_after.height;
        canvas_auxiliary.style.top = image_after.width;
        canvas_auxiliary.style.left = "2em";
        const canvas_auxiliary_after = document.getElementById("canvas_auxiliary_after");
        canvas_auxiliary_after.style.display = "";
        canvas_auxiliary_after.width = image_after.width;
        canvas_auxiliary_after.height = image_after.height;
        canvas_auxiliary_after.style.top = image_after.width;
        canvas_auxiliary_after.style.left = "2em";
        download_canvas.width = image_after.width;
        download_canvas.height = image_after.height;
        download_canvas.style.top = image_after.width;
        download_canvas.style.left = "2em";
        // img内容をcanvasに描画する
        const download_canvasCtx = download_canvas.getContext("2d");
        download_canvasCtx.drawImage(image_after, 0, 0, image_after.width, image_after.height);

        if (!poseLandmarker) {
            console.log("Wait for poseLandmarker to load before clicking!");
            return;
        }

        if (runningMode === "VIDEO") {
            runningMode = "IMAGE";
            poseLandmarker.setOptions({ runningMode: "IMAGE" });
        }
        poseLandmarker.detect(image_after, async (result) => {
            const canvas_afterCtx = canvas_after.getContext("2d");
            const drawingUtils_after = new DrawingUtils(canvas_afterCtx);
            const drawingUtils_download = new DrawingUtils(download_canvasCtx);


            for (const landmark of result.landmarks) {
                drawingUtils_after.drawLandmarks(landmark, {
                    radius: (data) => DrawingUtils.lerp(data.from?.z ?? 0, -0.15, 0.1, 5, 1),
                    // color: "orange",
                });
                drawingUtils_download.drawLandmarks(landmark, {
                    radius: (data) => DrawingUtils.lerp(data.from?.z ?? 0, -0.15, 0.1, 5, 1),
                    // color: "orange",
                });
                drawingUtils_after.drawConnectors(landmark, PoseLandmarker.POSE_CONNECTIONS);
                drawingUtils_download.drawConnectors(landmark, PoseLandmarker.POSE_CONNECTIONS);
            }
            if (result.landmarks.length === 0 || result.landmarks[0] == undefined || result.landmarks[0].length < 32) {
                // この後の処理を行わない
                console.log("landmarks is empty");
                return;
            }
            const left_ankle = result.landmarks[0][27];
            const right_ankle = result.landmarks[0][28];
            const ankle_center = {
                x: (left_ankle.x + right_ankle.x) / 2,
                y: (left_ankle.y + right_ankle.y) / 2,
                z: (left_ankle.z + right_ankle.z) / 2
            };
            console.log("ankle_center_after : ");
            console.log(ankle_center);
            // 鼻から足首の中点までの距離(身長に対応する長さ)を求める *Z座標は使わない
            const nose = result.landmarks[0][0];
            const nose_to_ankle_center_XY = Math.sqrt((nose.x - ankle_center.x) ** 2 + (nose.y - ankle_center.y) ** 2);




            if (result_before != [] && result_before.landmarks && result_before.landmarks[0].length > 32) {
                console.log("result_before.landmarks[0].length : ", result_before.landmarks[0].length);
                const canvas_after_overlayCtx = canvas_after_overlay.getContext("2d");
                const drawingUtils_after_overlay = new DrawingUtils(canvas_after_overlayCtx);
                for (const landmark of result_before.landmarks) {
                    // 足首の中点の座標を求める
                    const left_ankle_before = landmark[27];
                    const right_ankle_before = landmark[28];
                    const ankle_center_before = {
                        x: (left_ankle_before.x + right_ankle_before.x) / 2,
                        y: (left_ankle_before.y + right_ankle_before.y) / 2,
                        z: (left_ankle_before.z + right_ankle_before.z) / 2
                    };
                    console.log("ankle_center_before : ");
                    console.log(ankle_center_before);
                    // 鼻から足首の中点までの距離(身長に対応する長さ)を求める
                    const nose_before = landmark[0];
                    const nose_to_ankle_center_XY_before = Math.sqrt((nose_before.x - ankle_center_before.x) ** 2 + (nose_before.y - ankle_center_before.y) ** 2);
                    // 身長の比を求める
                    const height_ratio = nose_to_ankle_center_XY / nose_to_ankle_center_XY_before;
                    // before各座標の、足首の中点を基準にして、身長比をかけ、afterの座標に変換する
                    const landmark_from_ankle_center_before = landmark.map((point) => {
                        return {
                            x: (point.x - ankle_center_before.x) * height_ratio + ankle_center.x,
                            y: (point.y - ankle_center_before.y) * height_ratio + ankle_center.y,
                            z: (point.z - ankle_center_before.z) * height_ratio + ankle_center.z
                        };
                    });
                    // 変換後の座標を描画する
                    drawingUtils_after_overlay.drawLandmarks(landmark_from_ankle_center_before, {
                        radius: (data) => DrawingUtils.lerp(data.from?.z ?? 0, -0.15, 0.1, 5, 1),
                        lineWidth: 2,
                        color: "orange",
                    });
                    drawingUtils_after_overlay.drawConnectors(landmark_from_ankle_center_before, PoseLandmarker.POSE_CONNECTIONS,
                        { lineWidth: 2, color: "orange" });


                    // 正中線を描画する
                    const canvas_auxiliary_afterCtx = canvas_auxiliary_after.getContext("2d");
                    const drawingUtils_auxiliary_after = new DrawingUtils(canvas_auxiliary_afterCtx);

                    const shoulder_center = {
                        x: (result.landmarks[0][11].x + result.landmarks[0][12].x) / 2,
                        y: (result.landmarks[0][11].y + result.landmarks[0][12].y) / 2,
                        z: (result.landmarks[0][11].z + result.landmarks[0][12].z) / 2
                    }
                    const hip_center = {
                        x: (result.landmarks[0][23].x + result.landmarks[0][24].x) / 2,
                        y: (result.landmarks[0][23].y + result.landmarks[0][24].y) / 2,
                        z: (result.landmarks[0][23].z + result.landmarks[0][24].z) / 2
                    }
                    drawingUtils_auxiliary_after.drawLandmarks([ankle_center, hip_center, shoulder_center, result.landmarks[0][0]], {
                        radius: 3,
                    });
                    drawingUtils_auxiliary_after.drawConnectors([ankle_center, hip_center, shoulder_center, result.landmarks[0][0]], [{ start: 0, end: 1 }, { start: 1, end: 2 }, { start: 2, end: 3 }],
                        { lineWidth: 2 });

                    // 補助線を描画する
                    const canvas_auxiliaryCtx = canvas_auxiliary.getContext("2d");
                    const drawingUtils_auxiliary = new DrawingUtils(canvas_auxiliaryCtx);
                    const shoulder_center_before = {
                        x: (landmark_from_ankle_center_before[11].x + landmark_from_ankle_center_before[12].x) / 2,
                        y: (landmark_from_ankle_center_before[11].y + landmark_from_ankle_center_before[12].y) / 2,
                        z: (landmark_from_ankle_center_before[11].z + landmark_from_ankle_center_before[12].z) / 2
                    }
                    const hip_center_before = {
                        x: (landmark_from_ankle_center_before[23].x + landmark_from_ankle_center_before[24].x) / 2,
                        y: (landmark_from_ankle_center_before[23].y + landmark_from_ankle_center_before[24].y) / 2,
                        z: (landmark_from_ankle_center_before[23].z + landmark_from_ankle_center_before[24].z) / 2
                    }

                    drawingUtils_auxiliary.drawLandmarks([ankle_center, hip_center_before, shoulder_center_before, landmark_from_ankle_center_before[0]], {
                        radius: 3,
                        color: "orange",
                    });
                    drawingUtils_auxiliary.drawConnectors([ankle_center, hip_center_before, shoulder_center_before, landmark_from_ankle_center_before[0]], [{ start: 0, end: 1 }, { start: 1, end: 2 }, { start: 2, end: 3 }],
                        {
                            lineWidth: 2,
                            color: "orange",
                        });
                    console.log("POSE_CONNECTIONS : ");
                    console.log(PoseLandmarker.POSE_CONNECTIONS);


                    document.getElementById("checkbox_before").addEventListener("change", () => {
                        if (document.getElementById("checkbox_before").checked) {
                            canvas_after_overlay.style.opacity = 1;
                        } else {
                            canvas_after_overlay.style.opacity = 0;
                            canvas_auxiliary.style.opacity = 0;
                            document.getElementById("checkbox_auxiliary").checked = false;
                        }
                    });
                    document.getElementById("checkbox_after").addEventListener("change", () => {
                        if (document.getElementById("checkbox_after").checked) {
                            canvas_after.style.opacity = 1;
                        } else {
                            canvas_after.style.opacity = 0;
                            canvas_auxiliary_after.style.opacity = 0;
                            document.getElementById("checkbox_auxiliary_after").checked = false;
                        }
                    });
                    document.getElementById("checkbox_auxiliary").addEventListener("change", () => {
                        if (document.getElementById("checkbox_auxiliary").checked) {
                            canvas_auxiliary.style.opacity = 1;
                            canvas_before.style.opacity = 1;
                            document.getElementById("checkbox_before").checked = true;
                        } else {
                            canvas_auxiliary.style.opacity = 0;
                        }
                    });
                    document.getElementById("checkbox_auxiliary_after").addEventListener("change", () => {
                        if (document.getElementById("checkbox_auxiliary_after").checked) {
                            canvas_auxiliary_after.style.opacity = 1;
                            canvas_after.style.opacity = 1;
                            document.getElementById("checkbox_after").checked = true;
                        } else {
                            canvas_auxiliary_after.style.opacity = 0;
                        }
                    });


                    // 1秒おきに3回点滅させる
                    let count = 0;
                    const setLayerInterval = setInterval(() => {
                        count++;
                        if (count > 3) {
                            clearInterval(setLayerInterval);
                        }
                        canvas_after_overlay.style.opacity = 1 - canvas_after_overlay.style.opacity;
                        canvas_auxiliary.style.opacity = 1 - canvas_auxiliary.style.opacity;
                    }, 1000);

                    setTimeout(() => {
                        document.getElementById("checkboxes").style.display = "flex";
                        canvas_after_overlay.style.opacity = 0;
                        canvas_auxiliary.style.opacity = 0;
                        // ここにdownload_canvas_afterの設定を追加
                        let download_canvas_after = document.getElementById("download_canvas_after");
                        download_canvas_after.href = document.getElementById("download_canvas").toDataURL();
                        download_canvas_after.download = "pose_after.png";
                        download_canvas_after.style.display = "";

                        // ここに首関節の角度：Quaternion(x,y,z,w)を求める処理を追加
                        const nose_worldLandmarks = result.worldLandmarks[0][0];
                        const shoulder_center_worldLandmarks = {
                            x: (result.worldLandmarks[0][11].x + result.worldLandmarks[0][12].x) / 2,
                            y: (result.worldLandmarks[0][11].y + result.worldLandmarks[0][12].y) / 2,
                            z: (result.worldLandmarks[0][11].z + result.worldLandmarks[0][12].z) / 2
                        }
                        const hip_center_worldLandmarks = {
                            x: (result.worldLandmarks[0][23].x + result.worldLandmarks[0][24].x) / 2,
                            y: (result.worldLandmarks[0][23].y + result.worldLandmarks[0][24].y) / 2,
                            z: (result.worldLandmarks[0][23].z + result.worldLandmarks[0][24].z) / 2
                        }
                        // XY平面上での角度を求める。尻中央-肩中央のベクトルに対する、鼻-肩中央のベクトルの角度
                        const angle_XY_neck = Math.atan2(nose_worldLandmarks.y - shoulder_center_worldLandmarks.y, nose_worldLandmarks.x - shoulder_center_worldLandmarks.x) - Math.atan2(shoulder_center_worldLandmarks.y - hip_center_worldLandmarks.y, shoulder_center_worldLandmarks.x - hip_center_worldLandmarks.x);
                        // Quaternionを求める 小数点以下第7位で四捨五入する
                        const neckQuaternionXY = {
                            // テスト
                            x: 0,
                            y: 0,
                            // -1~1の範囲で、90度右に傾くと+0.5、左に傾くと-0.5になるようにする
                            z: -1 * Math.round(angle_XY_neck / Math.PI * 1000000) / 1000000,
                            w: 1,
                        };
                        const worldLandmarksTable = document.getElementById("worldLandmarksTable");
                        const neckTr = document.createElement("tr");
                        const neckTd1 = document.createElement("td");
                        const neckTdAngle = document.createElement("td");
                        const neckTd2 = document.createElement("td");
                        const neckTd3 = document.createElement("td");
                        const neckTd4 = document.createElement("td");
                        const neckTd5 = document.createElement("td");
                        neckTd1.innerText = "首(2)";
                        neckTdAngle.innerText = Math.round(angle_XY_neck / Math.PI * 1800) / 10;
                        neckTd2.innerText = neckQuaternionXY.x;
                        neckTd3.innerText = neckQuaternionXY.y;
                        neckTd4.innerText = neckQuaternionXY.z;
                        neckTd5.innerText = neckQuaternionXY.w;
                        neckTr.appendChild(neckTd1);
                        neckTr.appendChild(neckTdAngle);
                        neckTr.appendChild(neckTd2);
                        neckTr.appendChild(neckTd3);
                        neckTr.appendChild(neckTd4);
                        neckTr.appendChild(neckTd5);
                        worldLandmarksTable.appendChild(neckTr);

                        // ここに上半身の角度：Quaternion(x,y,z,w)を求める処理を追加
                        shoulder_center_worldLandmarks;
                        hip_center_worldLandmarks;
                        const ankle_center_worldLandmarks = {
                            x: (result.worldLandmarks[0][27].x + result.worldLandmarks[0][28].x) / 2,
                            y: (result.worldLandmarks[0][27].y + result.worldLandmarks[0][28].y) / 2,
                            z: (result.worldLandmarks[0][27].z + result.worldLandmarks[0][28].z) / 2
                        }
                        // XY平面上での角度を求める。足首中央-肩中央のベクトルに対する、肩中央-尻中央のベクトルの角度
                        const angle_XY_upper_body = Math.atan2(shoulder_center_worldLandmarks.y - hip_center_worldLandmarks.y, shoulder_center_worldLandmarks.x - hip_center_worldLandmarks.x) - Math.atan2(hip_center_worldLandmarks.y - ankle_center_worldLandmarks.y, hip_center_worldLandmarks.x - ankle_center_worldLandmarks.x);
                        // Quaternionを求める
                        const upperBodyQuaternionXY = {
                            x: 0,
                            y: 0,
                            // -0.7~+0.7の範囲で、右に90度傾く+0.7、左に90度傾く-0.7になるようにする
                            z: -1.4 * Math.round(angle_XY_upper_body / Math.PI * 1000000) / 1000000,
                            w: 1,
                        };
                        const upperBodyTr = document.createElement("tr");
                        const upperBodyTd1 = document.createElement("td");
                        const upperBodyTdAngle = document.createElement("td");
                        const upperBodyTd2 = document.createElement("td");
                        const upperBodyTd3 = document.createElement("td");
                        const upperBodyTd4 = document.createElement("td");
                        const upperBodyTd5 = document.createElement("td");
                        upperBodyTd1.innerText = "上半身(1)";
                        upperBodyTdAngle.innerText = Math.round(angle_XY_upper_body / Math.PI * 1800) / 10;
                        upperBodyTd2.innerText = upperBodyQuaternionXY.x;
                        upperBodyTd3.innerText = upperBodyQuaternionXY.y;
                        upperBodyTd4.innerText = upperBodyQuaternionXY.z;
                        upperBodyTd5.innerText = upperBodyQuaternionXY.w;
                        upperBodyTr.appendChild(upperBodyTd1);
                        upperBodyTr.appendChild(upperBodyTdAngle);
                        upperBodyTr.appendChild(upperBodyTd2);
                        upperBodyTr.appendChild(upperBodyTd3);
                        upperBodyTr.appendChild(upperBodyTd4);
                        upperBodyTr.appendChild(upperBodyTd5);
                        worldLandmarksTable.appendChild(upperBodyTr);

                        // ここに関節（左肘）の角度：Quaternion(x,y,z,w)を求める処理を追加
                        const left_shoulder = result.worldLandmarks[0][11];
                        const left_elbow = result.worldLandmarks[0][13];
                        const left_wrist = result.worldLandmarks[0][15];
                        // XY平面上での角度を求める
                        const angle_XY = Math.atan2(left_wrist.y - left_elbow.y, left_wrist.x - left_elbow.x) - Math.atan2(left_elbow.y - left_shoulder.y, left_elbow.x - left_shoulder.x);
                        // Quaternionを求める 小数点以下第7位で四捨五入する
                        const leftElbowQuaternionXY = {
                            x: 0,
                            y: 0,
                            // -1~1の範囲で、半周(180度)で1になるようにする
                            z: -1*Math.round(angle_XY / Math.PI * 1000000) / 1000000,
                            w: 1,
                        };
                        // const worldLandmarksTable = document.getElementById("worldLandmarksTable");
                        const leftElbowTr = document.createElement("tr");
                        const leftElbowTd1 = document.createElement("td");
                        const leftElbowTdAngle = document.createElement("td");
                        const leftElbowTd2 = document.createElement("td");
                        const leftElbowTd3 = document.createElement("td");
                        const leftElbowTd4 = document.createElement("td");
                        const leftElbowTd5 = document.createElement("td");
                        leftElbowTd1.innerText = "左肘(20)";
                        leftElbowTdAngle.innerText = Math.round(angle_XY / Math.PI * 1800) / 10;
                        leftElbowTd2.innerText = leftElbowQuaternionXY.x;
                        leftElbowTd3.innerText = leftElbowQuaternionXY.y;
                        leftElbowTd4.innerText = leftElbowQuaternionXY.z;
                        leftElbowTd5.innerText = leftElbowQuaternionXY.w;
                        leftElbowTr.appendChild(leftElbowTd1);
                        leftElbowTr.appendChild(leftElbowTdAngle);
                        leftElbowTr.appendChild(leftElbowTd2);
                        leftElbowTr.appendChild(leftElbowTd3);
                        leftElbowTr.appendChild(leftElbowTd4);
                        leftElbowTr.appendChild(leftElbowTd5);
                        worldLandmarksTable.appendChild(leftElbowTr);
                        // 同様に、関節（右肘）の角度：Quaternion(x,y,z,w)を求める処理を追加
                        const right_shoulder = result.worldLandmarks[0][12];
                        const right_elbow = result.worldLandmarks[0][14];
                        const right_wrist = result.worldLandmarks[0][16];
                        // XY平面上での角度を求める
                        const angle_XY_right = Math.atan2(right_wrist.y - right_elbow.y, right_wrist.x - right_elbow.x) - Math.atan2(right_elbow.y - right_shoulder.y, right_elbow.x - right_shoulder.x);
                        // Quaternionを求める
                        const rightElbowQuaternionXY = {
                            x: 0,
                            y: 0,
                            // 右側は左側の逆の角度になるので、-1をかける
                            z: Math.round(angle_XY_right / Math.PI * 1000000) / 1000000,
                            w: 1,
                        };
                        const rightElbowTr = document.createElement("tr");
                        const rightElbowTd1 = document.createElement("td");
                        const rightElbowTdAngle = document.createElement("td");
                        const rightElbowTd2 = document.createElement("td");
                        const rightElbowTd3 = document.createElement("td");
                        const rightElbowTd4 = document.createElement("td");
                        const rightElbowTd5 = document.createElement("td");
                        rightElbowTd1.innerText = "右肘(52)";
                        rightElbowTdAngle.innerText = Math.round(angle_XY_right / Math.PI * 1800) / 10;
                        rightElbowTd2.innerText = rightElbowQuaternionXY.x;
                        rightElbowTd3.innerText = rightElbowQuaternionXY.y;
                        rightElbowTd4.innerText = rightElbowQuaternionXY.z;
                        rightElbowTd5.innerText = rightElbowQuaternionXY.w;
                        rightElbowTr.appendChild(rightElbowTd1);
                        rightElbowTr.appendChild(rightElbowTdAngle);
                        rightElbowTr.appendChild(rightElbowTd2);
                        rightElbowTr.appendChild(rightElbowTd3);
                        rightElbowTr.appendChild(rightElbowTd4);
                        rightElbowTr.appendChild(rightElbowTd5);
                        worldLandmarksTable.appendChild(rightElbowTr);
                        // 関節（左肩）の角度：Quaternion(x,y,z,w)を求める処理を追加
                        // 今回はrightShoulder-leftShoulder-leftElbow間の角度を求める
                        const angle_XY_left_shoulder = Math.atan2(left_elbow.y - left_shoulder.y, left_elbow.x - left_shoulder.x) - Math.atan2(left_shoulder.y - right_shoulder.y, left_shoulder.x - right_shoulder.x);
                        // Quaternionを求める
                        const leftShoulderQuaternionXY = {
                            x: 0,
                            y: 0,
                            // 肩は0 -> 0.3 , 0.5*PI -> -0.4 になるように加工
                            z: Math.round((-1.4 * angle_XY_left_shoulder / Math.PI + 0.3) * 1000000) / 1000000,
                            w: 1,
                        };
                        const leftShoulderTr = document.createElement("tr");
                        const leftShoulderTd1 = document.createElement("td");
                        const leftShoulderTdAngle = document.createElement("td");
                        const leftShoulderTd2 = document.createElement("td");
                        const leftShoulderTd3 = document.createElement("td");
                        const leftShoulderTd4 = document.createElement("td");
                        const leftShoulderTd5 = document.createElement("td");
                        leftShoulderTd1.innerText = "左肩(18)";
                        leftShoulderTdAngle.innerText = Math.round(angle_XY_left_shoulder / Math.PI * 1800) / 10;
                        leftShoulderTd2.innerText = leftShoulderQuaternionXY.x;
                        leftShoulderTd3.innerText = leftShoulderQuaternionXY.y;
                        leftShoulderTd4.innerText = leftShoulderQuaternionXY.z;
                        leftShoulderTd5.innerText = leftShoulderQuaternionXY.w;
                        leftShoulderTr.appendChild(leftShoulderTd1);
                        leftShoulderTr.appendChild(leftShoulderTdAngle);
                        leftShoulderTr.appendChild(leftShoulderTd2);
                        leftShoulderTr.appendChild(leftShoulderTd3);
                        leftShoulderTr.appendChild(leftShoulderTd4);
                        leftShoulderTr.appendChild(leftShoulderTd5);
                        worldLandmarksTable.appendChild(leftShoulderTr);
                        // 関節（右肩）の角度：Quaternion(x,y,z,w)を求める処理を追加
                        // 今回はleftShoulder-rightShoulder-rightElbow間の角度を求める
                        let angle_XY_right_shoulder = Math.atan2(right_elbow.y - right_shoulder.y, right_elbow.x - right_shoulder.x) - Math.atan2(right_shoulder.y - left_shoulder.y, right_shoulder.x - left_shoulder.x);
                        if (angle_XY_right_shoulder > Math.PI)[
                            // 範囲を-PI~PIにする
                            angle_XY_right_shoulder -= 2 * Math.PI
                        ]
                        // Quaternionを求める
                        const rightShoulderQuaternionXY = {
                            x: 0,
                            y: 0,
                            // 右肩は、0 -> -0.3 , -0.5*PI ->0.4 になるように加工
                            z: -1 * Math.round((1.4 * angle_XY_right_shoulder / Math.PI + 0.3) * 1000000) / 1000000,
                            w: 1,
                        };
                        const rightShoulderTr = document.createElement("tr");
                        const rightShoulderTd1 = document.createElement("td");
                        const rightShoulderTdAngle = document.createElement("td");
                        const rightShoulderTd2 = document.createElement("td");
                        const rightShoulderTd3 = document.createElement("td");
                        const rightShoulderTd4 = document.createElement("td");
                        const rightShoulderTd5 = document.createElement("td");
                        rightShoulderTd1.innerText = "右肩(50)";
                        rightShoulderTdAngle.innerText = Math.round(angle_XY_right_shoulder / Math.PI * 1800) / 10;
                        rightShoulderTd2.innerText = rightShoulderQuaternionXY.x;
                        rightShoulderTd3.innerText = rightShoulderQuaternionXY.y;
                        rightShoulderTd4.innerText = rightShoulderQuaternionXY.z;
                        rightShoulderTd5.innerText = rightShoulderQuaternionXY.w;
                        rightShoulderTr.appendChild(rightShoulderTd1);
                        rightShoulderTr.appendChild(rightShoulderTdAngle);
                        rightShoulderTr.appendChild(rightShoulderTd2);
                        rightShoulderTr.appendChild(rightShoulderTd3);
                        rightShoulderTr.appendChild(rightShoulderTd4);
                        rightShoulderTr.appendChild(rightShoulderTd5);
                        worldLandmarksTable.appendChild(rightShoulderTr);


                    }, 4000);
                }
            }
            console.log("canvas_after_result : ");
            console.log(result);
            result_after = result;

        });

    };
});