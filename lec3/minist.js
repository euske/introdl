// minist.js
// by Yusuke Shinyama
// License: CC-SA-4.0

class MINIST {

  grid_x = 1.5;
  grid_y = 1.5;
  width = 24;
  height = 24;
  cellsize = 12;
  canvas = null;
  result = null;
  ctx = null;
  filling = null;
  prevpos = null;
  focus = {x:-1, y:-1};
  cell = null;
  out1 = null;
  out2 = null;

  setup(id1, id2) {
    this.canvas = document.getElementById(id1);
    this.result = document.getElementById(id2);
    this.ctx = this.canvas.getContext('2d');
    this.cell = Array(this.width * this.height);
    this.cell.fill(0);
    this.out1 = Array(100);
    this.out1.fill(0);
    this.out2 = Array(10);
    this.out2.fill(0);
    let obj = this;
    this.canvas.addEventListener('mousedown', (e) => {obj.mousedown(e)}, false);
    this.canvas.addEventListener('mouseup', (e) => {obj.mouseup(e)}, false);
    this.canvas.addEventListener('mousemove', (e) => {obj.mousemove(e)}, false);
    this.render();
  }

  clear() {
    this.cell.fill(0);
    this.update();
    this.render();
  }

  render() {
    let cs = this.cellsize;
    let ctx = this.ctx;
    ctx.fillStyle = 'white';
    ctx.fillRect(this.grid_x, this.grid_y, this.width*cs, this.height*cs);
    ctx.lineWidth = 1;
    for (let y = 0; y < this.height; y++) {
      for (let x = 0; x < this.width; x++) {
        let focused = (this.focus.x == x && this.focus.y == y);
        if (this.cell[this.width*y + x]) {
          ctx.fillStyle = 'black';
          ctx.fillRect(x*cs+this.grid_x, y*cs+this.grid_y, cs, cs);
          if (focused) {
            ctx.strokeStyle = 'white';
            ctx.strokeRect(x*cs+this.grid_x+1, y*cs+this.grid_y+1, cs-2, cs-2);
          }
        } else {
          ctx.strokeStyle = 'black';
          ctx.strokeRect(x*cs+this.grid_x, y*cs+this.grid_y, cs, cs);
          if (focused) {
            ctx.strokeStyle = 'black';
            ctx.strokeRect(x*cs+this.grid_x+1, y*cs+this.grid_y+1, cs-2, cs-2);
          }
        }
      }
    }
  }

  getpos(ev) {
    let b = this.canvas.getBoundingClientRect();
    return {x: ev.clientX-b.left, y:ev.clientY-b.top};
  }

  getcell(p) {
    let cs = this.cellsize;
    let x = Math.floor((p.x-this.grid_x)/cs);
    let y = Math.floor((p.y-this.grid_y)/cs);
    return {x:x, y:y};
  }

  isok(p) {
    return (0 <= p.x && p.x < this.width && 0 <= p.y && p.y < this.height);
  }

  setpix(x, y, filling) {
    this.cell[this.width*y + x] = this.filling;
    if (0 < x) {
      this.cell[this.width*y + x-1] = this.filling;
    }
    if (x+1 < this.width) {
      this.cell[this.width*y + x+1] = this.filling;
    }
    if (0 < y) {
      this.cell[this.width*(y-1) + x] = this.filling;
    }
    if (y+1 < this.height) {
      this.cell[this.width*(y+1) + x] = this.filling;
    }
  }

  mousedown(ev) {
    let p = this.getcell(this.getpos(ev));
    if (this.isok(p)) {
      this.filling = 1 - this.cell[this.width*p.y + p.x];
      this.prevpos = p;
      this.setpix(p.x, p.y, this.filling);
      this.update();
      this.render();
    }
  }

  mouseup(ev) {
    this.filling = null;
  }

  mousemove(ev) {
    let p = this.getcell(this.getpos(ev));
    if (this.filling !== null && this.isok(p)) {
      let dx = Math.abs(p.x - this.prevpos.x);
      let dy = Math.abs(p.y - this.prevpos.y);
      if (dx < dy) {
        for (let d = 0; d <= dy; d++) {
          let t = d/dy;
          let x = Math.floor(this.prevpos.x*(1-t) + p.x*t);
          let y = Math.floor(this.prevpos.y*(1-t) + p.y*t);
          this.setpix(x, y, this.filling);
        }
      } else {
        for (let d = 0; d <= dx; d++) {
          let t = d/dx;
          let x = Math.floor(this.prevpos.x*(1-t) + p.x*t);
          let y = Math.floor(this.prevpos.y*(1-t) + p.y*t);
          this.setpix(x, y, this.filling);
        }
      }
      this.prevpos = p;
      this.update();
    }
    this.focus = p;
    this.render();
  }

  update() {
    let cell = this.cell;
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
    //console.log(mi, y2);
    this.result.innerHTML = mi;
  }
}

var minist = new MINIST();
function minist_setup(id1, id2) {
  minist.setup(id1, id2);
}
function minist_clear() {
  minist.clear();
}
