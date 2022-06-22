// minist.js
// by Yusuke Shinyama
// License: CC-SA-4.0

function mkrgb(v) {
  v = Math.floor((1-v)*255);
  return 'rgb('+v+','+v+','+v+')';
}

class MINIST {

  input_x = 0.5;
  input_y = 130.5;
  input_cols = 24;
  input_rows = 24;
  input_cellsize = 10;
  out1_x = 320.5;
  out1_y = 0.5;
  out1_size = 100;
  out1_cellsize = 5;
  out2_x = 400.5;
  out2_y = 190.5;
  out2_size = 10;
  out2_cellsize = 12;

  canvas = null;
  result = null;
  ctx = null;

  cell = null;
  out1 = null;
  out2 = null;

  prevpos = null;
  focus = null;
  touchid = null;

  setup(id1, id2) {
    this.canvas = document.getElementById(id1);
    this.result = document.getElementById(id2);
    this.ctx = this.canvas.getContext('2d');
    this.cell = Array(this.input_cols * this.input_rows);
    this.out1 = Array(this.out1_size);
    this.out2 = Array(this.out2_size);
    let obj = this;
    this.canvas.addEventListener('mousedown', (e) => {obj.mousedown(e)}, false);
    this.canvas.addEventListener('mouseup', (e) => {obj.mouseup(e)}, false);
    this.canvas.addEventListener('mousemove', (e) => {obj.mousemove(e)}, false);
    this.canvas.addEventListener('touchstart', (e) => {obj.touchstart(e)}, false);
    this.canvas.addEventListener('touchend', (e) => {obj.touchend(e)}, false);
    this.canvas.addEventListener('touchmove', (e) => {obj.touchmove(e)}, false);
    this.clear();
  }

  clear() {
    this.cell.fill(0);
    this.focus = {t:null, x:-1, y:-1};
    this.update();
    this.render();
  }

  render() {
    let cs1 = this.input_cellsize;
    let cs2 = this.out1_cellsize;
    let cs3 = this.out2_cellsize;
    let ctx = this.ctx;
    let focus = this.focus;
    ctx.lineWidth = 1;
    ctx.fillStyle = 'white';
    ctx.fillRect(0, 0, this.canvas.width, this.canvas.height);
    for (let i = 0; i < this.out2_size; i++) {
      let focused = (focus.t === 'out2' && focus.x == 0 && focus.y == i);
      this.render1(this.out2_x, this.out2_y+cs3*i, cs3, this.out2[i], focused);
    }
    if (focus.t == 'out2') {
      let i = focus.y;
      let x0 = this.out2_x+cs3/2;
      let y0 = this.out2_y+cs3*i+cs3/2;
      let x1 = this.out1_x+cs2/2;
      let y1 = this.out1_y+cs2/2;
      for (let j = 0; j < this.out1_size; j++) {
        ctx.strokeStyle = mkrgb(MINIST_W2[i][j]);
        ctx.beginPath();
        ctx.moveTo(x0, y0);
        ctx.lineTo(x1, y1+cs2*j);
        ctx.stroke();
      }
    }
    for (let i = 0; i < this.out1_size; i++) {
      let focused = (focus.t === 'out1' && focus.x == 0 && focus.y == i);
      this.render1(this.out1_x, this.out1_y+cs2*i, cs2, this.out1[i], focused);
    }
    if (focus.t == 'input') {
      let i = this.input_cols*focus.y + focus.x;
      let x0 = this.input_x+cs1*focus.x+cs1/2;
      let y0 = this.input_y+cs1*focus.y+cs1/2;
      let x1 = this.out1_x+cs2/2;
      let y1 = this.out1_y+cs2/2;
      for (let j = 0; j < this.out1_size; j++) {
        ctx.strokeStyle = mkrgb(MINIST_W1[j][i]);
        ctx.beginPath();
        ctx.moveTo(x0, y0);
        ctx.lineTo(x1, y1+cs2*j);
        ctx.stroke();
      }
    }
    if (focus.t == 'out1') {
      let i = focus.y;
      let x0 = this.out1_x+cs2/2;
      let y0 = this.out1_y+cs2*i+cs2/2;
      let x1 = this.input_x+cs1/2;
      let y1 = this.input_y+cs1/2;
      for (let y = 0; y < this.input_rows; y++) {
        for (let x = 0; x < this.input_cols; x++) {
          let j = this.input_cols*y + x;
          ctx.strokeStyle = mkrgb(MINIST_W1[i][j]);
          ctx.beginPath();
          ctx.moveTo(x0, y0);
          ctx.lineTo(x1+cs1*x, y1+cs1*y);
          ctx.stroke();
        }
      }
      let x2 = this.out2_x+cs3/2;
      let y2 = this.out2_y+cs3/2;
      for (let j = 0; j < this.out2_size; j++) {
        ctx.strokeStyle = mkrgb(MINIST_W2[j][i]);
        ctx.beginPath();
        ctx.moveTo(x0, y0);
        ctx.lineTo(x2, y2+cs3*j);
        ctx.stroke();
      }
    }
    for (let y = 0; y < this.input_rows; y++) {
      for (let x = 0; x < this.input_cols; x++) {
        let focused = (focus.t === 'input' && focus.x == x && focus.y == y);
        this.render1(this.input_x+cs1*x, this.input_y+cs1*y, cs1,
                     this.cell[this.input_cols*y + x], focused);
      }
    }
  }

  render1(x, y, size, v, focused) {
    let ctx = this.ctx;
    ctx.fillStyle = mkrgb(v);
    ctx.fillRect(x, y, size, size);
    ctx.strokeStyle = 'black';
    ctx.strokeRect(x, y, size, size);
    if (focused) {
      ctx.strokeStyle = (0.5 < v)? 'white' : 'black';
      ctx.strokeRect(x+1, y+1, size-2, size-2);
    }
  }

  getpos(ev) {
    let b = this.canvas.getBoundingClientRect();
    let x = ev.clientX - b.left;
    let y = ev.clientY - b.top;
    let cs1 = this.input_cellsize;
    let cs2 = this.out1_cellsize;
    let cs3 = this.out2_cellsize;
    let cx = Math.floor((x-this.input_x)/cs1);
    let cy = Math.floor((y-this.input_y)/cs1);
    if (0 <= cx && cx < this.input_cols && 0 <= cy && cy < this.input_rows) {
      return { t:'input', x:cx, y:cy };
    }
    let i = Math.floor((y-this.out1_y)/cs2);
    if (this.out1_x <= x && x < this.out1_x+cs2 && 0 <= i && i < this.out1_size) {
      return { t:'out1', x:0, y:i };
    }
    i = Math.floor((y-this.out2_y)/cs3);
    if (this.out2_x <= x && x < this.out2_x+cs3 && 0 <= i && i < this.out2_size) {
      return { t:'out2', x:0, y:i };
    }
    return { t:null, x:-1, y:-1 };
  }

  setpix(x, y, v) {
    this.cell[this.input_cols*y + x] = v;
    if (0 < x) {
      this.cell[this.input_cols*y + x-1] = v;
    }
    if (x+1 < this.input_cols) {
      this.cell[this.input_cols*y + x+1] = v;
    }
    if (0 < y) {
      this.cell[this.input_cols*(y-1) + x] = v;
    }
    if (y+1 < this.input_rows) {
      this.cell[this.input_cols*(y+1) + x] = v;
    }
  }

  mousedown(ev) {
    let p = this.getpos(ev);
    if (p.t === 'input') {
      this.setpix(p.x, p.y, 1);
      this.prevpos = p;
      this.update();
      this.render();
    }
  }

  mouseup(ev) {
    this.prevpos = null;
  }

  mousemove(ev) {
    let p = this.getpos(ev);
    if (p.t === 'input' && this.prevpos !== null) {
      let dx = Math.abs(p.x - this.prevpos.x);
      let dy = Math.abs(p.y - this.prevpos.y);
      if (dx < dy) {
        for (let d = 0; d <= dy; d++) {
          let t = d/dy;
          let x = Math.floor(this.prevpos.x*(1-t) + p.x*t);
          let y = Math.floor(this.prevpos.y*(1-t) + p.y*t);
          this.setpix(x, y, 1);
        }
      } else {
        for (let d = 0; d <= dx; d++) {
          let t = d/dx;
          let x = Math.floor(this.prevpos.x*(1-t) + p.x*t);
          let y = Math.floor(this.prevpos.y*(1-t) + p.y*t);
          this.setpix(x, y, 1);
        }
      }
      this.prevpos = p;
      this.update();
    }
    this.focus = p;
    this.render();
  }

  touchstart(ev) {
    for (let t of ev.changedTouches) {
      if (this.touchid === null) {
        this.touchid = t.identifier;
        this.mousedown(t);
      }
    }
    ev.preventDefault();
  }

  touchend(ev) {
    for (let t of ev.changedTouches) {
      if (this.touchid !== null) {
        this.touchid = null;
        this.mouseup(t);
      }
    }
    ev.preventDefault();
  }

  touchmove(ev) {
    for (let t of ev.changedTouches) {
      if (this.touchid == t.identifier) {
        this.mousemove(t);
      }
    }
    ev.preventDefault();
  }

  update() {
    let cell = this.cell;
    let n = 0;
    for (let i = 0; i < cell.length; i++) {
      n += cell[i];
    }
    if (n == 0) {
      this.out1.fill(0);
      this.out2.fill(0);
      this.result.value = '???';
      return;
    }
    let out1 = this.out1;
    for (let i = 0; i < MINIST_W1.length; i++) {
      let v = MINIST_B1[i];
      for (let j = 0; j < cell.length; j++) {
        v += MINIST_W1[i][j] * cell[j];
      }
      out1[i] = 1/(1+Math.exp(-v));
    }
    let out2 = this.out2;
    let total = 0;
    for (let i = 0; i < MINIST_W2.length; i++) {
      let v = MINIST_B2[i];
      for (let j = 0; j < out1.length; j++) {
        v += MINIST_W2[i][j] * out1[j];
      }
      out2[i] = Math.exp(v);
      total += out2[i];
    }
    for (let i = 0; i < out2.length; i++) {
      out2[i] /= total;
    }
    let mi = 0;
    let my = -Infinity;
    for (let i = 0; i < out2.length; i++) {
      if (my < out2[i]) {
        mi = i; my = out2[i];
      }
    }
    //console.log(mi, out2);
    this.result.value = mi;
  }
}

var minist = new MINIST();
function minist_setup(id1, id2) {
  minist.setup(id1, id2);
}
function minist_clear() {
  minist.clear();
}
