import * as Three from 'three';
import { OrbitControls } from 'three/addons/controls/OrbitControls.js';
import { TransformControls } from 'three/addons/controls/TransformControls.js';
import "./style.css";

import init, { SimSolver } from "./wasm/wasm.js";

const wasm = await init();

const containerSize = 5;

const app = document.querySelector<HTMLDivElement>('#app')!;

const uiRoot = document.createElement("div");
uiRoot.style.position = "absolute";
uiRoot.style.left = "10px";
uiRoot.style.top = "10px";
uiRoot.style.zIndex = "10";
app.appendChild(uiRoot);

const info = document.createElement('div');
info.innerHTML = 'Press Space to add water blocks. <br>Click and drag red points to reshape them.';
info.style.color = 'white';
uiRoot.appendChild(info);

const methodPanel = document.createElement('div');
methodPanel.innerHTML = 'Method: <select id="methodSelect"><option value="sph">Smooth Particle Hydrodynamics (SPH)</option><option value="pbf">Position Based Fluids (PBF)</option></select>';
methodPanel.style.color = 'white';
uiRoot.appendChild(methodPanel);

const particleSizePanel = document.createElement('div');
particleSizePanel.innerHTML = 'Particle Size: <input type="number" id="particleSizeInput" value="0.15" step="0.01" min="0.01" max="1">';
particleSizePanel.style.color = 'white';
uiRoot.appendChild(particleSizePanel);

const gravityPanel = document.createElement('div');
gravityPanel.innerHTML = 'Gravity: <input type="number" id="gravityInput" value="1.0" step="0.1" min="0" max="20">';
gravityPanel.style.color = 'white';
uiRoot.appendChild(gravityPanel);

const pressureStiffnessPanel = document.createElement('div');
pressureStiffnessPanel.innerHTML = 'Pressure Stiffness: <input type="number" id="pressureStiffnessInput" value="1000.0" step="0.1" min="0" max="1000">';
pressureStiffnessPanel.style.color = 'white';
uiRoot.appendChild(pressureStiffnessPanel);

const visocityPanel = document.createElement('div');
visocityPanel.innerHTML = 'Viscosity: <input type="number" id="visocityInput" value="0.001" step="0.001" min="0" max="1">';
visocityPanel.style.color = 'white';
uiRoot.appendChild(visocityPanel);

const timeStepPanel = document.createElement('div');
timeStepPanel.innerHTML = 'Time Step: <input type="number" id="timeStepInput" value="0.01" step="0.001" min="0.001" max="1.0">';
timeStepPanel.style.color = 'white';
uiRoot.appendChild(timeStepPanel);

let isSimulating = false;
const runButton = document.createElement('button');
runButton.textContent = 'Run Simulation';
uiRoot.appendChild(runButton);

const renderer = new Three.WebGLRenderer({ antialias: true });
renderer.setSize(window.innerWidth, window.innerHeight);
app.appendChild(renderer.domElement);

const scene = new Three.Scene();
const camera = new Three.PerspectiveCamera(75, window.innerWidth / window.innerHeight, 0.1, 1000);
camera.position.z = 5;

scene.add(new Three.AmbientLight(0xffffff, 1.0));
const dirLight = new Three.DirectionalLight(0xffffff, 3.0);
dirLight.position.set(15, 20, 25);
scene.add(dirLight);

function addContainerMesh() {
  const boxGeometry = new Three.BoxGeometry(
    containerSize,
    containerSize,
    containerSize
  );

  const faceMaterial = new Three.MeshBasicMaterial({
    color: 0xaaaaaa,
    transparent: true,
    opacity: 0.05,
    depthWrite: false,
    side: Three.DoubleSide,
  });

  const faceMesh = new Three.Mesh(boxGeometry, faceMaterial);
  faceMesh.renderOrder = -1;
  scene.add(faceMesh);

  const edgeGeometry = new Three.EdgesGeometry(boxGeometry);
  const edgeMaterial = new Three.LineBasicMaterial({
    color: 0x666666,
  });

  const edgeMesh = new Three.LineSegments(edgeGeometry, edgeMaterial);
  edgeMesh.renderOrder = 1;
  scene.add(edgeMesh);
}
addContainerMesh();

const orbitControls = new OrbitControls(camera, renderer.domElement);

function resize() {
  const width = window.innerWidth;
  const height = window.innerHeight;
  renderer.setSize(width, height);
  camera.aspect = width / height;
  camera.updateProjectionMatrix();
}
window.addEventListener('resize', resize);

class Cuboid {
  center: Three.Vector3;
  size: Three.Vector3; // 中心からの距離

  constructor(center: Three.Vector3, size: Three.Vector3) {
    this.center = center;
    this.size = size;
  }

  negByIndex(v: Three.Vector3, index: number): Three.Vector3 {
    return new Three.Vector3(
      v.x * (index & 1 ? 1 : -1),
      v.y * (index & 2 ? 1 : -1),
      v.z * (index & 4 ? 1 : -1)
    );
  }

  getMeshPos(index: number): Three.Vector3 {
    const shift = this.negByIndex(this.size, index);
    return new Three.Vector3().addVectors(this.center, shift);
  }

  updateFromMesh(meshPos: Three.Vector3, index: number): void {
    // 反対の点を固定
    const prev = this.getMeshPos(index);
    const shift = new Three.Vector3().subVectors(meshPos, prev);
    shift.multiplyScalar(0.5);
    this.center.add(shift);
    this.size.add(this.negByIndex(shift, index));
  }
}

class Cuboids {
  cuboids: Map<number, Cuboid> = new Map();
  idCounter: number = 0;

  addCuboid(center: Three.Vector3, size: Three.Vector3): number {
    const id = this.idCounter++;
    this.cuboids.set(id, new Cuboid(center, size));
    return id;
  }

  removeCuboid(id: number): void {
    this.cuboids.delete(id);
  }

  updateCuboid(id: number, meshPos: Three.Vector3, index: number): void {
    const cuboid = this.cuboids.get(id);
    if (cuboid) {
      cuboid.updateFromMesh(meshPos, index);
    }
  }

  getParticles(particleSize: number): Three.Vector3[] {
    const posSet = new Set<string>();
    for (const cuboid of this.cuboids.values()) {
      const minX = Math.ceil((cuboid.center.x - cuboid.size.x) / particleSize);
      const maxX = Math.floor((cuboid.center.x + cuboid.size.x) / particleSize);
      const minY = Math.ceil((cuboid.center.y - cuboid.size.y) / particleSize);
      const maxY = Math.floor((cuboid.center.y + cuboid.size.y) / particleSize);
      const minZ = Math.ceil((cuboid.center.z - cuboid.size.z) / particleSize);
      const maxZ = Math.floor((cuboid.center.z + cuboid.size.z) / particleSize);

      for (let x = minX; x <= maxX; x++) {
        for (let y = minY; y <= maxY; y++) {
          for (let z = minZ; z <= maxZ; z++) {
            if (Math.abs(x) > containerSize / particleSize / 2 ||
              Math.abs(y) > containerSize / particleSize / 2 ||
              Math.abs(z) > containerSize / particleSize / 2) {
              continue;
            }
            const posKey = `${x},${y},${z}`;
            posSet.add(posKey);
          }
        }
      }
    }
    return Array.from(posSet).map((posKey) => {
      const [x, y, z] = posKey.split(',').map(Number);
      return new Three.Vector3(x * particleSize, y * particleSize, z * particleSize);
    });
  }
}

const cuboids = new Cuboids();

class PickHelper {
  raycaster: Three.Raycaster;
  selected = false;
  transformControls: TransformControls;
  meshMap: Map<number, Three.Mesh> = new Map();

  settingPanel: HTMLDivElement | null = null;

  constructor() {
    this.raycaster = new Three.Raycaster();

    this.transformControls = new TransformControls(camera, renderer.domElement);
    scene.add(this.transformControls.getHelper());
    this.transformControls.addEventListener('dragging-changed', (event) => {
      orbitControls.enabled = !event.value;
    });
    this.transformControls.addEventListener('objectChange', () => {
      if (this.selected) {
        const id = this.transformControls.object!.userData.id;
        const index = this.transformControls.object!.userData.index;
        const mesh = this.meshMap.get(id * 9 + index)!;
        this.updateCuboid(id, mesh.position, index);
      }
    });
  }

  updateCuboid(id: number, meshPos: Three.Vector3, index: number): void {
    cuboids.updateCuboid(id, meshPos, index);
    const cuboid = cuboids.cuboids.get(id)!;
    for (let i = 0; i < 8; i++) {
      const meshPos = cuboid.getMeshPos(i);
      const mesh = this.meshMap.get(id * 9 + i)!;
      mesh.position.copy(meshPos);
    }
    const boxMesh = this.meshMap.get(id * 9 + 8)!;
    boxMesh.position.copy(cuboid.center);
    boxMesh.scale.set(cuboid.size.x * 2, cuboid.size.y * 2, cuboid.size.z * 2);
  }

  addCuboid(center: Three.Vector3, size: Three.Vector3): void {
    const id = cuboids.addCuboid(center, size);
    const cuboid = cuboids.cuboids.get(id)!;
    for (let i = 0; i < 8; i++) {
      const meshPos = cuboid.getMeshPos(i);
      const mesh = new Three.Mesh(
        new Three.SphereGeometry(0.05, 16, 16),
        new Three.MeshStandardMaterial({ color: 0xff0000 })
      );
      mesh.position.copy(meshPos);
      mesh.userData.id = id;
      mesh.userData.index = i;
      scene.add(mesh);
      this.meshMap.set(id * 9 + i, mesh);
    }
    const mesh = new Three.Mesh(
      new Three.BoxGeometry(1, 1, 1),
      new Three.MeshStandardMaterial({ color: 0x0000ff, transparent: true, opacity: 0.5, emissive: 0x0000ff })
    );
    mesh.position.copy(cuboid.center);
    mesh.scale.set(cuboid.size.x * 2, cuboid.size.y * 2, cuboid.size.z * 2);
    mesh.userData.id = id;
    mesh.userData.index = 8;
    scene.add(mesh);
    this.meshMap.set(id * 9 + 8, mesh);
  }

  removeControlPoint(id: number): void {
    for (let i = 0; i < 9; i++) {
      const mesh = this.meshMap.get(id * 9 + i);
      if (mesh) {
        scene.remove(mesh);
        mesh.geometry.dispose();
        (mesh.material as Three.Material).dispose();
        this.meshMap.delete(id * 9 + i);
      }
    }
    cuboids.removeCuboid(id);
  }

  unselect() {
    if (this.selected) {
      this.selected = false;
      this.transformControls.detach();

      if (this.settingPanel) {
        this.settingPanel.remove();
        this.settingPanel = null;
      }
    }
  }

  pick(mouse: { x: number; y: number }, camera: Three.Camera) {
    this.raycaster.setFromCamera(new Three.Vector2(mouse.x, mouse.y), camera);
    const intersects = this.raycaster.intersectObjects(Array.from(this.meshMap.values()));

    this.unselect();

    if (intersects.length > 0) {
      const mesh = intersects[0].object as Three.Mesh;
      const index = mesh.userData.index;
      if (index === 8) {
        return;
      }
      this.selected = true;
      this.transformControls.attach(mesh);
      this.createSettingPanel();
    }
  }

  createSettingPanel() {
    if (!this.selected) return;

    this.settingPanel = document.createElement('div');
    this.settingPanel.style.padding = "6px";
    this.settingPanel.style.background = "rgba(0, 0, 0, 0.5)";
    this.settingPanel.style.borderRadius = "4px";

    const id = this.transformControls.object!.userData.id;

    const removeButton = document.createElement('button');
    removeButton.textContent = 'Remove';
    removeButton.style.marginLeft = '10px';
    this.settingPanel.appendChild(removeButton);

    removeButton.addEventListener('click', () => {
      if (this.selected) {
        this.removeControlPoint(id);
        this.unselect();
      }
    });

    uiRoot.appendChild(this.settingPanel);
  }

  hide() {
    for (const mesh of this.meshMap.values()) {
      mesh.visible = false;
    }
    this.unselect();
  }

  show() {
    for (const mesh of this.meshMap.values()) {
      mesh.visible = true;
    }
  }
}

const pickHelper = new PickHelper();
window.addEventListener('keydown', (event) => {
  if (event.code === 'Space' && !isSimulating) {
    const center = new Three.Vector3(0, 0, 0);
    const size = new Three.Vector3(1, 1, 1);
    pickHelper.addCuboid(center, size);
  }
});

class PointerHelper {
  pointerDownPos: { x: number; y: number } | null = null;
  isDrag = false;

  constructor() {
    renderer.domElement.addEventListener('pointerdown', this.onPointerDown.bind(this));
    renderer.domElement.addEventListener('pointermove', this.onPointerMove.bind(this));
    renderer.domElement.addEventListener('pointerup', (event) => this.onPointerUp(event, camera));
  }

  onPointerDown(event: PointerEvent) {
    this.pointerDownPos = { x: event.clientX, y: event.clientY };
    this.isDrag = false;
  }

  onPointerMove(event: PointerEvent) {
    if (this.pointerDownPos) {
      const dx = event.clientX - this.pointerDownPos.x;
      const dy = event.clientY - this.pointerDownPos.y;
      if (Math.sqrt(dx * dx + dy * dy) > 5) {
        this.isDrag = true;
      }
    }
  }

  onPointerUp(event: PointerEvent, camera: Three.Camera) {
    if (!this.isDrag && this.pointerDownPos) {
      const rect = renderer.domElement.getBoundingClientRect();
      const mouse = {
        x: ((event.clientX - rect.left) / rect.width) * 2 - 1,
        y: -((event.clientY - rect.top) / rect.height) * 2 + 1
      };
      pickHelper.pick(mouse, camera);
    }
    this.pointerDownPos = null;
    this.isDrag = false;
  }
}

new PointerHelper();

class Simulation {
  geometry: Three.BufferGeometry;
  pointsMesh: Three.Points;
  attr: Three.BufferAttribute;

  solver: SimSolver;

  constructor(particles: Three.Vector3[], particleSize: number, gravity: number, pressureStiffness: number, visocity: number) {
    const n = particles.length;
    this.solver = new SimSolver(n, particleSize, containerSize, gravity, pressureStiffness, visocity);

    const positions = new Float32Array(wasm.memory.buffer, this.solver.pos(), n * 3);
    for (const [i, p] of particles.entries()) {
      positions[i * 3] = p.x;
      positions[i * 3 + 1] = p.y;
      positions[i * 3 + 2] = p.z;
    }

    this.geometry = new Three.BufferGeometry();
    this.attr = new Three.BufferAttribute(positions, 3);
    this.geometry.setAttribute('position', this.attr);
    this.attr.setUsage(Three.DynamicDrawUsage);
    this.geometry.setAttribute('position', this.attr);

    const material = new Three.PointsMaterial({ color: 0x00aaff, size: 0.5 * particleSize });
    this.pointsMesh = new Three.Points(this.geometry, material);
  }

}

let simulation: Simulation | null = null;

function startSimulation() {
  const particleSizeInput = document.getElementById('particleSizeInput') as HTMLInputElement;
  const particleSize = parseFloat(particleSizeInput.value);
  const particles = cuboids.getParticles(particleSize);

  const gravity = parseFloat((document.getElementById('gravityInput') as HTMLInputElement).value);
  const pressureStiffness = parseFloat((document.getElementById('pressureStiffnessInput') as HTMLInputElement).value);
  const visocity = parseFloat((document.getElementById('visocityInput') as HTMLInputElement).value);

  simulation = new Simulation(particles, particleSize, gravity, pressureStiffness, visocity);
  scene.add(simulation.pointsMesh);
}

function stopSimulation() {
  if (simulation) {
    scene.remove(simulation.pointsMesh);
    simulation.geometry.dispose();
    (simulation.pointsMesh.material as Three.Material).dispose();
    simulation = null;
  }
}

runButton.addEventListener('click', () => {
  isSimulating = !isSimulating;
  runButton.textContent = isSimulating ? 'Stop Simulation' : 'Run Simulation';
  if (isSimulating) {
    pickHelper.hide();
    startSimulation();
  } else {
    pickHelper.show();
    stopSimulation();
  }
});

function animate() {
  requestAnimationFrame(animate);
  if (simulation) {
    simulation.attr.needsUpdate = true;
    const timeStep = parseFloat((document.getElementById('timeStepInput') as HTMLInputElement).value);
    const method = (document.getElementById('methodSelect') as HTMLSelectElement).value;
    if (method === 'sph') {
      simulation.solver.step_sph(timeStep);
    } else {
      simulation.solver.step_pbf(timeStep);
    }
  }
  renderer.render(scene, camera)
} 
animate();