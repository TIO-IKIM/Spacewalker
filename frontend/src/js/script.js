/*
GUI for project "Spacewalker"
Authors: Fabian Hörst, Lukas Heine, Gijs Luijten, Miriam Balzer
 */

import * as THREE from 'three';
import { OrbitControls } from 'three/examples/jsm/controls/OrbitControls';
import { Pane } from 'tweakpane';

// Storage
const Minio = require('minio');
let minioClient = new Minio.Client({
    endPoint: '0.0.0.0',
    port: 9000,
    accessKey: 'demo',
    secretKey: 'demodemo',
    useSSL: false,
});
const minioBucket = "spacewalker-projects";

// Data
let data2d = window.data2d;
let data3d = window.data3d;

// create deep copies
let old_data2d;
let old_data3d;
updateBuffer();

let labels = []
window.existing_labels.forEach(function(element) {
    labels.push(element['fields'])
})
//  Constants
let scale = 5;
let rollOverScale = 1;
const color = new THREE.Color();

// Global display variables
let renderer;
let scene2D;
let scene3D;
let camera2D;
let camera3D;
let controls2D;
let controls3D;
let mesh2D;
let mesh3D;
let rollOverMesh2D;
let rollOverMesh3D;
let gridHelper2D;
let gridHelper3D;
let axesHelper2D;
let axesHelper3D;
let axesHelper2Dplane = new THREE.Plane(new THREE.Vector3(0, 1, 0), 0);

// raycast
let raycaster3D;
let raycaster2D;
const mouse = new THREE.Vector2( 1, 1 );
let paintColor = 0xffffff //0xff00000;
let activeLabel = 0;
let multiSelect = false;
let cursorIn3D = false;
let isMouseDown = false;
let hoverId;

//query points
let octahedron2d;
let octahedron3d;

// slider
let sliderPos = window.innerWidth / 4;

// GUI / Menu Pane
let pane;
let progressField;
let classparams;
// tooltip
let tooltip_template = document.createRange().createContextualFragment(`
    <div id="tooltip" class="noselect" style="display: none; position: absolute; pointer-events: auto; font-size: 14px; width: 300px; text-align: left; padding: 12px; background: #f8f8f8; box-shadow: 0 2px 4px rgba(0, 0, 0, 0.1); font-family: 'Arial', sans-serif; border-radius: 8px;">
      <div id="point_tip" style="font-weight: bold; margin-bottom: 8px;"></div>
      <div id="group_tip" style="padding: 4px;">
          <img id="tooltipImage" style="width: 100%; height: auto; border-radius: 4px; display: none;" src="" alt="Thumbnail">
          <div id="tooltipText" style="display: none; max-height: 150px; overflow-y: auto;"></div>
      </div>
    </div>`);
document.body.appendChild(tooltip_template);

document.body.style.overflow = 'hidden'; // Disable overflow to prevent scrolling
    
let tooltip_state = { display: "none" }
let $tooltip = document.querySelector('#tooltip');
let $point_tip = document.querySelector('#point_tip');
let tooltipImage = document.getElementById('tooltipImage');

$tooltip.addEventListener('wheel', function(event) {
    event.preventDefault();
    event.stopPropagation();
}, { passive: false });

let maxCoordinate = 0;
// find largest point
data2d.forEach(point => {
    maxCoordinate = Math.max(maxCoordinate, Math.abs(point.x), Math.abs(point.y));
});
data3d.forEach(point => {
    maxCoordinate = Math.max(maxCoordinate, Math.abs(point.x), Math.abs(point.y), Math.abs(point.z));
});
let layoutScale = maxCoordinate * scale * 2;

// gallery pane
let isMouseInsidePanel = false;
let menuHeight = 300;

function updateBuffer(){
    // update buffer
    old_data2d = JSON.parse(JSON.stringify(data2d));
    old_data3d = JSON.parse(JSON.stringify(data3d));
}

function KeyPress(e) {
    var evtobj = window.event? event : e
    if (evtobj.keyCode == 90 && evtobj.ctrlKey) {
        data2d = JSON.parse(JSON.stringify(old_data2d));
        data3d = JSON.parse(JSON.stringify(old_data3d));
        initMeshes();
    };
}
document.onkeydown = KeyPress;


function getProgress(){
let count = data2d.filter(item => item.cluster_id !== 0).length;
return `${count} / ${data2d.length} (${Math.round((count/data2d.length)*100)} %)`;
}

// Initialize and run the application
init();

document.addEventListener('mousemove', (event) => {
    if (event.clientX < sliderPos) {
        controls2D.enabled = true;
        controls3D.enabled = false; 
        cursorIn3D = false;
    } else {
        controls2D.enabled = false;
        controls3D.enabled = true;
        cursorIn3D = true;
    }
});
document.addEventListener('contextmenu', (event) => {
    event.preventDefault();
  });
document.addEventListener( 'mousemove', onMouseMove );
document.addEventListener( 'mousedown', onRMBClick );
document.addEventListener( 'mouseup', onRMBRelease );

document.getElementById('text2d').addEventListener('click', function() {
    // Move slider to the far right (100% - slider's width to account for its size)
    document.getElementById('slider').style.left = `calc(100% - ${0.5*document.getElementById('slider').offsetWidth}px)`;
    sliderPos = window.innerWidth;
});

document.getElementById('text3d').addEventListener('click', function() {
    // Move slider to the far right (100% - slider's width to account for its size)
    document.getElementById('slider').style.left = `calc(0% - ${0.5*document.getElementById('slider').offsetWidth}px)`;
    sliderPos = 0;
});

window.addEventListener('resize', function() {
// Resize for 2D
camera2D.aspect = window.innerWidth / window.innerHeight;
camera2D.updateProjectionMatrix();

// Resize for 3D
camera3D.aspect = window.innerWidth / window.innerHeight;
camera3D.updateProjectionMatrix();

// Resize renderer
renderer.setSize(window.innerWidth, window.innerHeight);


});
renderer.setAnimationLoop(animate);
// Functions
function init() {

    // Create a Scene
    scene2D = new THREE.Scene();
    scene2D.background = new THREE.Color( 0x404040 );
    scene3D = new THREE.Scene();
    scene3D.background = new THREE.Color( 0x161616 );
    
    // init renderer
    initRenderer();
    
    // init slider
    initSlider();

    // init scene settings like cameras, controls, grids, meshes, and  lights
    initCameras();
    initControls();
    initGrids();
    initLights();
    initMeshes();
    initRaycaster();

    // init GUI
    initMenu();
}

function initMenu() {
    // Define pane
    pane = new Pane({
        title: 'Settings',
        expanded: true
    });

    initClassSelectorGUI();
    initCursorGUI();
    initClassSettingsGUI();
}

function initClassSelectorGUI() {
    // Class settings for selecting class
}

function initClassSettingsGUI() {
    const folder_add_class = pane.addFolder({
        title: 'Class selection',
        expanded: true
    });

    // Create dropdown menu options for 'Name' field
    let nameOptions = {};
    labels.forEach(label => {
        nameOptions[label.name] = label.name;
    });

    // Initialize class parameters
    classparams = {
        Name: labels[0].name, // Set default name
        ClassID: labels[0].class_id, // Set default class ID
        Color: labels[0].color, // Set default color
        Progress: getProgress(),
    };

    paintColor = Number("0x" + labels[0].color.slice(1));
    activeLabel = Number(labels[0].class_id);

    // Add fields to the folder
    let nameField = folder_add_class.addBinding(classparams, 'Name', {
        options: nameOptions
    });

    let classIDField = folder_add_class.addBinding(classparams, 'ClassID', {
        disabled: true // Disable editing for Class ID
    });

    let colorField = folder_add_class.addBinding(classparams, 'Color', {
        disabled: true // Disable editing for Color
    });

    progressField = folder_add_class.addBinding(classparams, 'Progress', {
        disabled: true // Disable editing for Color
    });

        // Event listener for name field changes
    nameField.on('change', value => {
    // Find corresponding label based on selected name
    let selectedLabel = labels.find(label => label.name === value.value);
    // Update Class ID and Color fields based on the selected label
    classparams.ClassID = selectedLabel.class_id;
    classparams.Color = selectedLabel.color;
    // Update the fields in the GUI
    classIDField.refresh();
    colorField.refresh();
    // Update the paintColor variable
    paintColor = Number("0x" + selectedLabel.color.slice(1));
    activeLabel = Number(selectedLabel.class_id);
    initMeshes();
    });

    let saveAnnotations = pane.addButton({
        title: "Save annotations",
        disabled: false
    });

    saveAnnotations.on("click", () => handleSaveAnnotationsClick());

    document.addEventListener('keydown', (event) => {
        const key = event.key;
        if (!isNaN(key) && key >= 0 && key <= 9) {
            const index = parseInt(key);
            if (index < labels.length) {
                const selectedLabel = labels[index];
                classparams.Name = selectedLabel.name;
                classparams.ClassID = selectedLabel.class_id;
                classparams.Color = selectedLabel.color;
                // Update the fields in the GUI
                nameField.refresh();
                classIDField.refresh();
                colorField.refresh();
                // Update the paintColor variable
                paintColor = Number("0x" + selectedLabel.color.slice(1));
                activeLabel = Number(selectedLabel.class_id);
                // update color of rollovermeshes
                scene2D.remove(rollOverMesh2D);
                scene3D.remove(rollOverMesh3D);
                rollOverMesh2D.material.color.set(paintColor);
                rollOverMesh3D.material.color.set(paintColor);
                scene2D.add(rollOverMesh2D);
                scene3D.add(rollOverMesh3D);
            }
        }
    });
}


function initCursorGUI() {
    let cursorparams = {
        Scale: 1,
        Multiselect: false,
        Point_scaling: 5,
        color_2D: '#404040',
        color_3D: '#161616',
    };

    // Class settings for selecting class
    const folder_annotation = pane.addFolder({
        title: 'Annotation Settings',
        expanded: true
    });

    // Scaling of the Raycast bowl
    folder_annotation.addBinding(cursorparams, 'Scale', {
        step: 1,
        speed: 5,
    }).on("change", (ev) => {
        rollOverScale = parseInt(ev.value);
        rollOverMesh2D.scale.setScalar(rollOverScale);
        rollOverMesh3D.scale.setScalar(rollOverScale);
    });

    // Activating multiselect
    folder_annotation.addBinding(cursorparams, 'Multiselect').on("change", (ev) => {
        multiSelect = ev.value;
    });

    folder_annotation.addBinding(cursorparams, 'Point_scaling', {
        speed: 5,
    }).on("change", (ev) => {
        scale = ev.value;
        initMeshes();
        // update grids

        layoutScale = maxCoordinate * scale * 2;

        // update the query point

        // init grids
        initGrids();
    });

    folder_annotation.addBinding(cursorparams, 'color_2D').on("change", (ev) => {
        scene2D.background = new THREE.Color(ev.value);
    });

    folder_annotation.addBinding(cursorparams, 'color_3D').on("change", (ev) => {
        scene3D.background = new THREE.Color(ev.value);
    });
}

// Three JS initialization
function initCameras() {
    camera2D = new THREE.OrthographicCamera(
        window.innerWidth / - 2,
        window.innerWidth / 2 ,
        window.innerHeight / 2,
        window.innerHeight / - 2,
        1,
        1000,
    );
    //camera2D.position.set(layoutScale, layoutScale, layoutScale);
    camera2D.position.set(0, 20, 0);
    camera2D.lookAt(0, 0, 0 );

    camera3D = new THREE.PerspectiveCamera(
        60,
        window.innerWidth / window.innerHeight,
        0.1,
        1000
    );
    camera3D.position.set(layoutScale, layoutScale, layoutScale);
    camera3D.lookAt( 0, 0, 0 );

}

function initControls() {
    controls2D = new OrbitControls( camera2D, renderer.domElement );
    controls2D.enableRotate = false;
    controls2D.mouseButtons = {
        LEFT: THREE.MOUSE.ROTATE,
        MIDDLE: THREE.MOUSE.PAN,
    };
    controls2D.update();

    controls3D = new OrbitControls( camera3D, renderer.domElement );
    controls3D.mouseButtons = {
        LEFT: THREE.MOUSE.ROTATE,
        MIDDLE: THREE.MOUSE.PAN,
    };
    controls3D.update();
}

function initGrids() {
    scene2D.remove( gridHelper2D );
    scene2D.remove( axesHelper2D );
    // 2D grid
    gridHelper2D = new THREE.GridHelper(layoutScale, layoutScale / 10);
    scene2D.add( gridHelper2D );
    axesHelper2D = new THREE.AxesHelper( layoutScale / 2 );
    scene2D.add( axesHelper2D );

    scene3D.remove( gridHelper3D );
    scene3D.remove( axesHelper3D );
    // 3D grid
    gridHelper3D = new THREE.GridHelper(layoutScale, layoutScale / 10);
    scene3D.add( gridHelper3D );
    axesHelper3D = new THREE.AxesHelper( layoutScale / 2 );
    scene3D.add( axesHelper3D );
}

function initLights() {
    let light2D = new THREE.HemisphereLight(
        0xffffff, // bright sky color
        0x888888, // dim ground color
        3 // intensity
    );
    light2D.position.set( 0, 1, 0 );
    scene2D.add( light2D );

    let light3D = new THREE.HemisphereLight(
        0xffffff, // bright sky color
        0x888888, // dim ground color
        3 // intensity
    );
    light3D.position.set( 0, 1, 0 );
    scene3D.add( light3D );
}

function initRaycaster() {
    raycaster2D = new THREE.Raycaster();
    raycaster3D = new THREE.Raycaster();
}

function initRenderer() {
    renderer = new THREE.WebGLRenderer( { antialias: true } );
    renderer.setSize(window.innerWidth, window.innerHeight);
    renderer.setPixelRatio( window.devicePixelRatio );
    document.body.appendChild(renderer.domElement);
}

// Slider
function initSlider() {
    const slider = document.querySelector('#slider');

    /**
     * Event handler for the pointerdown event.
     * Disables 3D and 2D controls and adds event listeners for pointermove and pointerup events.
     */
    function onPointerDown(e) {
        if (e.isPrimary === false) return;
        controls3D.enabled = false;
        controls2D.enabled = false;
        window.addEventListener('pointermove', onPointerMove);
        window.addEventListener('pointerup', onPointerUp);
    }

    /**
     * Event handler for the pointerup event.
     * Enables 3D controls and removes event listeners for pointermove and pointerup events.
     */
    function onPointerUp() {
        controls3D.enabled = true;
        controls2D.enabled = false;
        window.removeEventListener('pointermove', onPointerMove);
        window.removeEventListener('pointerup', onPointerUp);
    }

    /**
     * Event handler for the pointermove event.
     * Updates the position of the slider based on the pointer's x-coordinate.
     */
    function onPointerMove(e) {
        if (e.isPrimary === false) return;
        sliderPos = Math.max(0, Math.min(window.innerWidth, e.pageX));
        slider.style.left = sliderPos - (slider.offsetWidth / 2) + 'px';
        // text3D.style.left = sliderPos + 80 + 'px';
    }

    slider.style.touchAction = 'none';
    slider.addEventListener('pointerdown', onPointerDown);
}

function initMeshes() {
    scene2D.remove(mesh2D);
    scene2D.remove(rollOverMesh2D);
    scene3D.remove(mesh3D);
    scene3D.remove(rollOverMesh3D);

    // 2D Mesh
    let geometry2D = new THREE.IcosahedronGeometry(0.5, 3);
    let material2D = new THREE.MeshPhongMaterial({ color: 0xffffff });
    mesh2D = new THREE.InstancedMesh(geometry2D, material2D, data2d.length);
    let matrix2D = new THREE.Matrix4();
    data2d.forEach(function (element, idx) {
        matrix2D.setPosition(element.x * scale, element.z * scale, element.y * scale);
        mesh2D.setMatrixAt(idx, matrix2D);
        let labelColor = labels.find(item => item.class_id === element.cluster_id)['color'];
        labelColor = new THREE.Color( labelColor )
        mesh2D.setColorAt(idx, labelColor);
    }); 
    mesh2D.receiveShadow = true;
    scene2D.add(mesh2D);
    
    // 2D selector mesh
    let selectorGeometry2D = new THREE.SphereGeometry(1);
    let selectorMaterial2D = new THREE.MeshBasicMaterial({ color: paintColor, opacity: 0.2, transparent: true });
    rollOverMesh2D = new THREE.Mesh(selectorGeometry2D, selectorMaterial2D);
    rollOverMesh2D.scale.setScalar(rollOverScale);
    scene2D.add(rollOverMesh2D);

    //3D Mesh
    let geometry3D = new THREE.IcosahedronGeometry(0.5, 3);
    let material3D = new THREE.MeshPhongMaterial({ color: 0xffffff });
    mesh3D = new THREE.InstancedMesh(geometry3D, material3D, data3d.length);
    
    let matrix3D = new THREE.Matrix4();
    data3d.forEach(function (element, idx) {
        matrix3D.setPosition(element.x * scale, element.y * scale, element.z * scale);
        mesh3D.setMatrixAt(idx, matrix3D);

        let labelColor = labels.find(item => item.class_id === element.cluster_id)['color'];
        labelColor = new THREE.Color( labelColor )
        mesh3D.setColorAt(idx, labelColor);
    });
    mesh3D.receiveShadow = true;
    scene3D.add(mesh3D);

    // 3D selector mesh
    let selectorGeometry3D = new THREE.SphereGeometry(1);
    let selectorMaterial3D = new THREE.MeshBasicMaterial({ color: paintColor, opacity: 0.2, transparent: true });
    rollOverMesh3D = new THREE.Mesh(selectorGeometry3D, selectorMaterial3D);
    rollOverMesh3D.scale.setScalar(rollOverScale);
    scene3D.add(rollOverMesh3D);
}


// rendering 
function render() {
    // Render 2D scene (left)
    renderer.setScissor(0, 0, window.innerWidth, window.innerHeight);
    renderer.setScissorTest(true);
    renderer.clear();
    renderer.render(scene2D, camera2D);

    // Clear depth buffer
    renderer.clearDepth();

    // Render 3D scene (right)
    renderer.setScissor(sliderPos, 0, window.innerWidth, window.innerHeight);
    renderer.setScissorTest(true);
    renderer.render(scene3D, camera3D);
}

// mouse movement/controls
function onMouseMove( event ) {
    event.preventDefault();
    mouse.x = ( event.clientX / window.innerWidth ) * 2 - 1;
    mouse.y = - ( event.clientY / window.innerHeight ) * 2 + 1;
}
function onRMBClick( event ) {
    if ( event.button === 2 ) {
        event.preventDefault();
        updateBuffer();
        isMouseDown = true;
    }
    else {
        isMouseDown = false;
    }
}
function onRMBRelease( event ) {
    if ( event.button === 2 ) {
        event.preventDefault();
        isMouseDown = false;
    }
    else {
        isMouseDown = false;
    }
}


function animate() {
    if (cursorIn3D) {
        controls3D.update();
        raycaster3D.setFromCamera( mouse, camera3D );
        const intersection3D = raycaster3D.intersectObject( mesh3D );

        if ((intersection3D.length > 0)){
            const instanceId = intersection3D[0].instanceId;
            const mouse_position = [mouse.x, mouse.y]

            rollOverMesh3D.position.set(intersection3D[0].point.x, intersection3D[0].point.y, intersection3D[0].point.z)
            showTooltip(mouse_position, instanceId);
            // necessary: mouse3D position, datum
            if (isMouseDown && !isMouseInsidePanel){
                if (multiSelect){
                for (let i = 0; i < mesh3D.count; i++) {
                    const matrix = new THREE.Matrix4();
                    mesh3D.getMatrixAt(i, matrix);

                    const position = new THREE.Vector3();
                    position.setFromMatrixPosition(matrix);

                    if (position.distanceTo(rollOverMesh3D.position) < rollOverMesh3D.scale.x) {
                        mesh3D.setColorAt( i, color.setHex( paintColor ) );
                        mesh2D.setColorAt( i, color.setHex( paintColor ) );

                        data3d[i]['cluster_id'] = activeLabel;
                        data2d[i]['cluster_id'] = activeLabel;

                        mesh3D.instanceColor.needsUpdate = true;
                        mesh2D.instanceColor.needsUpdate = true;
                    }
                }
                }
            else {
                    mesh3D.setColorAt( instanceId, color.setHex( paintColor ) );
                    mesh2D.setColorAt( instanceId, color.setHex( paintColor ) );

                    data3d[instanceId]['cluster_id'] = activeLabel;
                    data2d[instanceId]['cluster_id'] = activeLabel;

                    mesh3D.instanceColor.needsUpdate = true;
                    mesh2D.instanceColor.needsUpdate = true;
                }
            }
        }
    }
    else {
        controls2D.update();
        raycaster2D.setFromCamera( mouse, camera2D );

        // Intersection with the 2D plane
        const planeIntersection = new THREE.Vector3();
        if (raycaster2D.ray.intersectPlane(axesHelper2Dplane, planeIntersection)) {
            rollOverMesh2D.position.set(planeIntersection.x, planeIntersection.y, planeIntersection.z);
            if (isMouseDown && !isMouseInsidePanel){
                if (multiSelect){
                for (let i = 0; i < mesh2D.count; i++) {
                    const matrix = new THREE.Matrix4();
                    mesh2D.getMatrixAt(i, matrix);

                    const position = new THREE.Vector3();
                    position.setFromMatrixPosition(matrix);

                    if (position.distanceTo(rollOverMesh2D.position) < rollOverMesh2D.scale.x) {
                        mesh2D.setColorAt( i, color.setHex( paintColor ) );
                        mesh3D.setColorAt( i, color.setHex( paintColor ) );
                        
                        data3d[i]['cluster_id'] = activeLabel;
                        data2d[i]['cluster_id'] = activeLabel;

                        mesh2D.instanceColor.needsUpdate = true;
                        mesh3D.instanceColor.needsUpdate = true;
                    }
                }
                }
            }
        }
        const intersection2D = raycaster2D.intersectObject( mesh2D );

        if ((intersection2D.length > 0)){
            const instanceId = intersection2D[0].instanceId;
            const mouse_position = [mouse.x, mouse.y]

            rollOverMesh2D.position.set(intersection2D[0].point.x, intersection2D[0].point.y, intersection2D[0].point.z)
            showTooltip(mouse_position, instanceId);
            if (isMouseDown && !isMouseInsidePanel){
                if (multiSelect){
                for (let i = 0; i < mesh2D.count; i++) {
                    const matrix = new THREE.Matrix4();
                    mesh2D.getMatrixAt(i, matrix);

                    const position = new THREE.Vector3();
                    position.setFromMatrixPosition(matrix);

                    if (position.distanceTo(rollOverMesh2D.position) < rollOverMesh2D.scale.x) {
                        mesh2D.setColorAt( i, color.setHex( paintColor ) );
                        mesh3D.setColorAt( i, color.setHex( paintColor ) );
                        
                        data3d[i]['cluster_id'] = activeLabel;
                        data2d[i]['cluster_id'] = activeLabel;

                        mesh2D.instanceColor.needsUpdate = true;
                        mesh3D.instanceColor.needsUpdate = true;
                    }
                }
                }
            else {
                    mesh2D.setColorAt( instanceId, color.setHex( paintColor ) );
                    mesh3D.setColorAt( instanceId, color.setHex( paintColor ) );

                    data3d[instanceId]['cluster_id'] = activeLabel;
                    data2d[instanceId]['cluster_id'] = activeLabel;

                    mesh2D.instanceColor.needsUpdate = true;
                    mesh3D.instanceColor.needsUpdate = true;
                }
            }
        }
    }
    // update progress
    classparams.Progress = getProgress();
    progressField.value = getProgress();
    progressField.refresh();
    render();
}

// tooltip
function updateTooltip() {
    $tooltip.style.display = tooltip_state.display;
    $tooltip.style.left = tooltip_state.left + 'px';
    $tooltip.style.top = tooltip_state.top + 'px';
    $point_tip.innerText = tooltip_state.name;

    if (tooltip_state.type === 'image') {
        tooltipImage.src = tooltip_state.content;
        tooltipImage.style.display = 'block';
        tooltipText.style.display = 'none';
    } else if (tooltip_state.type === 'text') {
        tooltipText.innerText = tooltip_state.content;
        tooltipText.style.display = 'block';
        tooltipImage.style.display = 'none';
    }
}

function showTooltip(mouse_position, instanceId) {
    let datapoint = data3d[instanceId];
    let file_path = datapoint["preview"];

    if (hoverId !== instanceId) {
        hoverId = instanceId;

        loadMinioData(minioBucket, file_path)
            .then((result) => {
                tooltip_state.display = "block";
                tooltip_state.name = "Showing Object " + instanceId;
                tooltip_state.content = result.content;
                tooltip_state.type = result.type;

                updateTooltip();
            })
            .catch((error) => {
                console.error('Error loading Minio data:', error);
            });
    }
}

function loadMinioData(bucketName, objectName) {
    return new Promise((resolve, reject) => {
        minioClient.getObject(bucketName, objectName, function (err, dataStream) {
            if (err) {
                console.error(err);
                reject(err);
                return;
            }

            let chunks = [];

            dataStream.on('data', function (chunk) {
                chunks.push(chunk);
            });

            dataStream.on('end', function () {
                let fileData = Buffer.concat(chunks);

                if (objectName.endsWith('.png') || objectName.endsWith('.jpg') || objectName.endsWith('.jpeg')) {
                    let base64Image = fileData.toString('base64');
                    let dataUrl = 'data:image/png;base64,' + base64Image;
                    resolve({ type: 'image', content: dataUrl });
                } else if (objectName.endsWith('.txt')) {
                    let textContent = fileData.toString('utf8');
                    resolve({ type: 'text', content: textContent });
                } else {
                    reject(new Error('Unsupported file type'));
                }
            });

            dataStream.on('error', function (err) {
                console.error(err);
                reject(err);
            });
        });
    });
}


function handleSaveAnnotationsClick() {
    const csrfToken = document.getElementsByName('csrfmiddlewaretoken')[0].value;
    // Maybe consider just passing the labels instead of the whole json
    // We only have to return data2d, since we only need the combination of filename and class once
    fetch('/gui/', {
        method: 'POST',
        headers: {
          'X-CSRFToken': csrfToken,
          'Content-Type': 'application/json'
        },
        body: JSON.stringify(data2d)
      })
      .then(response => {
        if (!response.ok) {
          throw new Error('Network response was not ok');
        }
        return response.json();
      })
      .then(data => {
        alert('Saved!')
      })
      .catch(error => {
        console.error('There was a problem with the request:', error);
      });
}

document.getElementById('inputForm').addEventListener('submit', function(event) {
    event.preventDefault();
    const formData = new FormData();
    formData.append('requestType', 'query')
    formData.append('project', data2d[0]['project_name'])
    formData.append('model', data2d[0]['model'])

    const textInput = document.getElementById('textInput').value;
    const imageInput = document.getElementById('imageInput').files[0];

    if (textInput) {
        formData.append('textInput', textInput);
    }

    if (imageInput) {
        const reader = new FileReader();
        reader.onload = function(e) {
            formData.append('imageInput', imageInput);
            sendRequest(formData);
        };
        reader.readAsDataURL(imageInput);
    } else {
        sendRequest(formData);
    }

function sendRequest(formData) {
    const csrfToken = document.getElementsByName('csrfmiddlewaretoken')[0].value;
    fetch('/gui/', {
        method: 'POST',
        body: formData,
        headers: {
            'X-CSRFToken': csrfToken
        }
    })
    .then(response => response.json())
    .then(data => {
        // Handle success
        scene2D.remove(octahedron2d);
        scene3D.remove(octahedron3d);

        const queryPoint2d = data['2d_embedding'];
        const queryPoint3d = data['3d_embedding'];

        const queryGeometry = new THREE.OctahedronGeometry(1);
        const material = new THREE.MeshBasicMaterial({ color: 0xff0000 }); // red color (optional)
        
        octahedron2d = new THREE.Mesh(queryGeometry, material);
        octahedron3d = new THREE.Mesh(queryGeometry, material);

        octahedron2d.position.set(queryPoint2d[0] * scale, 0, queryPoint2d[1] * scale);
        octahedron3d.position.set(queryPoint3d[0] * scale, queryPoint3d[1] * scale, queryPoint3d[2] * scale);

        scene2D.add(octahedron2d);
        scene3D.add(octahedron3d);

        //2D
        camera2D.position.set(queryPoint2d[0] * scale, camera2D.position.y, queryPoint2d[1] * scale);
        camera2D.lookAt(queryPoint2d[0] * scale, 0, queryPoint2d[1] * scale);

        controls2D.target.set(queryPoint2d[0] * scale, 0, queryPoint2d[1] * scale); // Reset the target to the center or desired point.
        controls2D.update();

        //3D
        camera3D.lookAt(queryPoint3d[0] * scale, queryPoint3d[1] * scale, queryPoint3d[2] * scale);

        controls3D.target.set(queryPoint3d[0] * scale, queryPoint3d[1] * scale, queryPoint3d[2] * scale); // Reset the target to the center or desired point.
        controls3D.update();

        render();
    })
    .catch(error => {
        console.error('Error:', error);
        // Handle error
    });
}
});

document.addEventListener("keydown", function(event) {
    if (event.key === "q") {
        const activeElement = document.activeElement;
        if (activeElement === document.getElementById("textInput") || activeElement === document.getElementById("imageInput")) {
          // Do nothing if the focus is on the text or file input fields
          return;
        }
      const panel = document.getElementById("collapsible-panel");
      const panelContent = document.getElementById("panel-content");
      panelContent.innerHTML = ''; // Sets the HTML content of the panel

      if (panel.style.height === "0px" || panel.style.height === "") {
        panel.style.height = `${menuHeight}px`; // Adjust to your preferred height
      } else {
        panel.style.height = "0";
        isMouseInsidePanel = false;
      }
    }
  });

  let hoveredIndices = []; // Store hovered indices

document.addEventListener("keydown", function(event) {
    const activeElement = document.activeElement;
  
    // If focused on an input field (text or image), allow typing 'q' and don't trigger the panel
    if (activeElement === document.getElementById("textInput") || activeElement === document.getElementById("imageInput")) {
      if (event.key === "q") {
        // Allow typing 'q' in the input field, and do nothing else
        return;
      }
    }
  
    // Handle the 'q' key press for opening the panel when focus is not on input fields
    if (event.key === "q" && activeElement !== document.getElementById("textInput") && activeElement !== document.getElementById("imageInput")) {
    //   event.preventDefault(); // Prevent the default action for the 'q' key
  
      hoveredIndices = [];  // Reset indices each time 'q' is pressed
  
      // Determine if we're in 3D or 2D context
      if (cursorIn3D) {
          // Add all points covered by the rollOverMesh3D
          for (let i = 0; i < mesh3D.count; i++) {
            const matrix = new THREE.Matrix4();
            mesh3D.getMatrixAt(i, matrix);
  
            const position = new THREE.Vector3();
            position.setFromMatrixPosition(matrix);
  
            // Check if the point is within the radius of the rollOverMesh
            if (position.distanceTo(rollOverMesh3D.position) < rollOverMesh3D.scale.x) {
              hoveredIndices.push(i); // Add index of the covered point
            }
        }
      } else {
        raycaster2D.setFromCamera(mouse, camera2D);
          // Add all points covered by the rollOverMesh2D
          for (let i = 0; i < mesh2D.count; i++) {
            const matrix = new THREE.Matrix4();
            mesh2D.getMatrixAt(i, matrix);
  
            const position = new THREE.Vector3();
            position.setFromMatrixPosition(matrix);
  
            // Check if the point is within the radius of the rollOverMesh
            if (position.distanceTo(rollOverMesh2D.position) < rollOverMesh2D.scale.x) {
              hoveredIndices.push(i); // Add index of the covered point
            }
        }
      }
  
      hoveredIndices.forEach((index) => {
        // const objectName = data2d[index]["file_reference"].replace(".upload", "");  // You should define this function or have a list of object names.
        const objectName = data2d[index].preview;
        let color = new THREE.Color();

        mesh2D.getColorAt(index, color);

        loadMinioData(minioBucket, objectName)
          .then((data) => {
            if (data.type === 'image') {
              let imgElement = document.createElement('img');
              imgElement.src = data.content; // Base64-encoded image daqta
              imgElement.style.margin = '5px';
              imgElement.style.border = `5px solid #${color.getHexString()}`; // Add a red border/frame to the image
              imgElement.style.borderRadius = '8px'; // Optional: rounded corners for the border
            //   imgElement.style.boxShadow = '1px 1px 1px 1px';

              imgElement.setAttribute('data-index', index);

                // Add a contextmenu event listener to the image (right-click)
              imgElement.addEventListener('contextmenu', (e) => {
              e.preventDefault(); // Prevent the default right-click menu
              const originalIndex = e.target.getAttribute('data-index'); // Get the index of the image
              imgElement.style.border = `5px solid #${paintColor.toString(16)}`;
                
              const new_color = new THREE.Color()
              mesh3D.setColorAt( originalIndex, new_color.setHex( paintColor ) );
              mesh2D.setColorAt( originalIndex, new_color.setHex( paintColor ) );

              data3d[originalIndex]['cluster_id'] = activeLabel;
              data2d[originalIndex]['cluster_id'] = activeLabel;

              mesh3D.instanceColor.needsUpdate = true;
              mesh2D.instanceColor.needsUpdate = true;
            });

              const panelContent = document.getElementById('panel-content');
              panelContent.appendChild(imgElement);
            }
          })
          .catch((error) => {
            console.error('Error loading MinIO data:', error);
          });
      });
    }
  });

  // Resizing
const panel = document.getElementById("collapsible-panel");
const resizeHandle = document.getElementById("resize-handle");

let isResizing = false;

function startResizing(e) {
  // Only start resizing on left-click (button 0)
  if (e.button !== 0) return;

  isResizing = true;
  document.body.style.cursor = "ns-resize";

  // Attach mousemove and mouseup listeners for resizing only
  window.addEventListener("mousemove", resizePanel);
  window.addEventListener("mouseup", stopResizing);
}

function resizePanel(e) {
  if (!isResizing) return;

  // Calculate the new height
  menuHeight = window.innerHeight - e.clientY;
  panel.style.height = `${menuHeight}px`;
}

function stopResizing(e) {
  isResizing = false;
  document.body.style.cursor = "";

  // Remove the listeners to prevent interference
  window.removeEventListener("mousemove", resizePanel);
  window.removeEventListener("mouseup", stopResizing);
}

// Prevent right-click from starting or interfering with resizing
resizeHandle.addEventListener("contextmenu", (e) => {
  e.preventDefault(); // Prevent default context menu
  stopResizing(); // Ensure resizing stops on right-click
});

// Start resizing only with a left-click on the resize handle
resizeHandle.addEventListener("mousedown", startResizing);

panel.addEventListener("mouseenter", () => {
    isMouseInsidePanel = true;
  });
  
  // Event listener for when the mouse leaves the panel
  panel.addEventListener("mouseleave", () => {
    isMouseInsidePanel = false;
  });

  document.getElementById('close-button').addEventListener('click', function() {
    document.getElementById('collapsible-panel').style.height = '0';
});