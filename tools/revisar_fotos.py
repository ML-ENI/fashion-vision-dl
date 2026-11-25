"""Galería local de revisión. Solo escribe el registro pendiente de decisiones.

No ejecuta CLIP ni publica datasets. Se inicia expresamente desde el cuaderno
revision_visual_interactiva.ipynb o con python tools/revisar_fotos.py.
"""
from __future__ import annotations

import csv
import gzip
import hashlib
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
import json
from pathlib import Path
import secrets
import tempfile
import threading
from urllib.parse import parse_qs, urlparse
import webbrowser

COLUMNS = ['article_id', 'file_sha256', 'decision', 'reason']
DECISIONS = {'keep', 'exclude_detail', 'uncertain'}


def read_csv(path):
    if not path.exists():
        return []
    opener = gzip.open if path.suffix == '.gz' else open
    with opener(path, 'rt', encoding='utf-8-sig', newline='') as stream:
        return list(csv.DictReader(stream))


class ReviewStore:
    def __init__(self, root):
        self.root = Path(root).resolve()
        self.output = self.root / 'data/processed'
        pointer = json.loads((self.output / 'CURRENT.json').read_text(encoding='utf-8'))
        run = (self.output / pointer['run_dir']).resolve()
        if not run.is_relative_to(self.output) or not run.is_dir():
            raise ValueError('Publicación local no válida')
        self.run_name = run.name
        manifest_path = run / 'image_manifest.csv.gz'
        checks = json.loads((run / 'checksums.json').read_text(encoding='utf-8'))
        if hashlib.sha256(manifest_path.read_bytes()).hexdigest() != checks['image_manifest.csv.gz']:
            raise ValueError('El manifiesto ha cambiado: comprobar 02 antes de revisar')
        self.rows = {}
        for row in read_csv(manifest_path):
            if (row.get('assigned_split') in ['train', 'validation', 'test']
                    and row.get('index_group_name') != 'Baby/Children'):
                article = row['article_id']
                if article in self.rows:
                    raise ValueError('Identificador repetido en el manifiesto')
                self.rows[article] = row
        self.path = self.root / 'config/visual_review_pending_chat.csv'
        self.manual_path = self.root / 'config/training_visual_review.csv'
        self.lock = threading.RLock()
        self._load()

    def _load(self):
        # Releer antes de guardar para conservar decisiones añadidas desde otro proceso.
        self.pending = self._index(self.path)
        self.manual = self._index(self.manual_path)

    def _index(self, path):
        result = {}
        for row in read_csv(path):
            if set(row) != set(COLUMNS) or row['article_id'] in result or row['decision'] not in DECISIONS:
                raise ValueError(f'Registro con formato o decisiones no válidas: {path.name}')
            result[row['article_id']] = row
        return result

    def status(self, row):
        saved = self.pending.get(row['article_id']) or self.manual.get(row['article_id'])
        if saved and saved['file_sha256'] != row['file_sha256']:
            raise ValueError(f"La foto revisada ha cambiado: {row['article_id']}")
        return saved

    def batch(self, split='train', label='', mode='clip_uncertain', after=''):
        if split not in ['train', 'validation', 'test'] or mode not in ['clip_uncertain', 'pending', 'uncertain', 'all']:
            raise ValueError('Filtro desconocido')
        with self.lock:
            self._load()
            candidates = []
            for article, row in sorted(self.rows.items()):
                if row['assigned_split'] != split or (label and row['label'] != label):
                    continue
                saved = self.status(row)
                if mode == 'clip_uncertain' and (saved or row.get('auto_visual_decision') != 'uncertain'
                                                  or row.get('split') != split):
                    continue
                if mode == 'pending' and saved:
                    continue
                if mode == 'uncertain' and (not saved or saved['decision'] != 'uncertain'):
                    continue
                candidates.append((article, row, saved))
            remaining = [(a, r, s) for a, r, s in candidates if a > after]
            selected = remaining[:12]
            return {
                'cards': [{'article_id': a, 'label': r['label'],
                           'automatic': r.get('auto_visual_decision', ''),
                           'decision': s['decision'] if s else '',
                           'reason': s['reason'] if s else ''} for a, r, s in selected],
                'after': selected[-1][0] if selected else after,
                'more': len(remaining) > 12, 'total': len(candidates),
                'remaining': len(remaining), 'saved': len(self.pending), 'run': self.run_name,
                'labels': sorted({r['label'] for r in self.rows.values()}),
            }

    def save(self, article, decision, reason=''):
        if decision not in DECISIONS or article not in self.rows or len(reason) > 1000:
            raise ValueError('Decisión o identificador no válido')
        defaults = {'keep': 'Prenda reconocible: conservar. Revisión humana en galería.',
                    'exclude_detail': 'Encuadre de detalle o parcial: excluir. Revisión humana en galería.',
                    'uncertain': 'Revisión humana: duda pendiente de resolver.'}
        with self.lock:
            self._load()
            row = self.rows[article]
            self.status(row)
            self.pending[article] = dict(article_id=article, file_sha256=row['file_sha256'],
                                         decision=decision, reason=reason.strip() or defaults[decision])
            self.path.parent.mkdir(parents=True, exist_ok=True)
            temporary = None
            try:
                with tempfile.NamedTemporaryFile('w', encoding='utf-8', newline='', dir=self.path.parent,
                                                 prefix='.visual-review-', suffix='.pending', delete=False) as stream:
                    temporary = Path(stream.name)
                    writer = csv.DictWriter(stream, fieldnames=COLUMNS, lineterminator='\n')
                    writer.writeheader()
                    writer.writerows(self.pending.values())
                temporary.replace(self.path)
            finally:
                if temporary and temporary.exists():
                    temporary.unlink()
            return {'saved': True, 'article_id': article, 'decision': decision,
                    'reason': self.pending[article]['reason']}

    def image(self, article):
        row = self.rows[article]
        path = (self.output / row['image_path']).resolve()
        if not path.is_relative_to(self.output):
            raise ValueError('Ruta de fotografía no válida')
        data = path.read_bytes()
        if hashlib.sha256(data).hexdigest() != row['prepared_sha256']:
            raise ValueError('La fotografía preparada ha cambiado')
        return data


PAGE = r'''<!doctype html><html lang="es"><meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>Revisar fotografías · DL</title><style>
:root{font-family:system-ui,sans-serif;color:#17212b;background:#f3f6fa;font-size:15px}
body{margin:0 auto;max-width:1250px;padding:22px}h1{color:#00458a;font-size:26px;margin:0 0 10px}
.note{background:white;border-left:4px solid #f18d01;padding:12px;line-height:1.5}
nav{display:flex;flex-wrap:wrap;gap:12px;align-items:end;margin:18px 0}label{display:grid;gap:5px}
select,button,textarea{font:inherit}select,button{padding:8px;border:1px solid #bdcadd;border-radius:6px;background:white}
button{cursor:pointer}button:disabled{opacity:.5;cursor:wait}.primary{background:#00458a;color:white}
#grid{display:grid;grid-template-columns:repeat(auto-fit,minmax(235px,1fr));gap:14px}
.card{background:white;border:2px solid #dce4ee;border-radius:9px;padding:12px;text-align:center}
.card[data-decision="keep"]{border-color:#00458a}.card[data-decision="exclude_detail"]{border-color:#f18d01}
.card[data-decision="uncertain"]{border-style:dashed;border-color:#8c97a8}
.photo{width:170px;height:180px;object-fit:contain;background:white;display:block;margin:4px auto 10px}
.actions{display:flex;gap:5px;justify-content:center;margin:10px 0}.actions button{font-size:13px;padding:8px 7px}
.actions button[aria-pressed="true"]{outline:2px solid #00458a;font-weight:700}
.id,.auto{color:#536477;font-size:12px;margin-top:5px}.status{font-size:13px;min-height:20px;color:#00458a}
textarea{box-sizing:border-box;width:100%;height:52px;font-size:12px;padding:6px;resize:vertical;border:1px solid #dce4ee}
#error{color:#a02020;white-space:pre-wrap}footer{margin:20px 0;display:flex;gap:12px;align-items:center}
</style><h1>Revisar encuadres de prendas</h1>
<div class="note"><b>Conservar:</b> se reconoce la prenda. <b>Excluir:</b> zoom o encuadre parcial inadecuado.
<b>Duda:</b> prefieres decidir después. Cada clic se guarda automáticamente en tu ordenador.
Puedes cambiar una decisión pulsando otro botón. Baby/Children queda fuera de esta revisión.<br>
Las marcas se guardan como <b>revisión pendiente</b>; todavía no cambian el dataset ni ejecutan CLIP.
En Garment Set, anota si solo aparece parte del conjunto: es distinto de un zoom.</div>
<nav><label>Partición<select id="split"><option value="train">Train · empezar aquí</option>
<option value="validation">Validation</option><option value="test">Test · solo encuadre</option></select></label>
<label>Tipo de prenda<select id="label"><option value="">Todas</option></select></label>
<label>Mostrar<select id="mode"><option value="clip_uncertain">Dudas de CLIP conservadas · sin revisar</option>
<option value="uncertain">Mis dudas guardadas</option><option value="pending">Todas las fotos sin revisar</option>
<option value="all">Todas · corregir decisiones</option></select></label>
<button id="reset">Volver al principio</button></nav><p id="info">Cargando…</p><p id="error" role="alert"></p>
<div id="grid"></div><footer><button class="primary" id="next">Siguiente lote</button>
<span>Revisa las que quieras. Cerrar la página no borra lo guardado.</span></footer>
<script>
let cursor='',nextCursor='',busy=0,lastBatch=null;
const el=id=>document.getElementById(id), names={keep:'Conservar',exclude_detail:'Excluir',uncertain:'Duda'};
async function api(path,options){const r=await fetch(path,options);if(!r.ok)throw Error(await r.text());return r.json()}
function error(e){el('error').textContent=e.message}
async function mark(card,item,decision,reason){
 busy++;el('next').disabled=true;card.querySelectorAll('button').forEach(b=>b.disabled=true);
 const status=card.querySelector('.status');status.textContent='Guardando…';
 try{const result=await api('decision',{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify({article_id:item.article_id,decision,reason})});
 item.decision=decision;item.reason=result.reason;card.querySelector('textarea').value=result.reason;
 card.querySelector('textarea').dataset.dirty='0';card.dataset.decision=decision;status.textContent='Guardado · '+names[decision];
 card.querySelectorAll('button[data-value]').forEach(b=>b.setAttribute('aria-pressed',b.dataset.value===decision));
 }catch(e){status.textContent='NO guardado. Vuelve a pulsar el botón.';error(e)}
 finally{busy--;card.querySelectorAll('button').forEach(b=>b.disabled=false);el('next').disabled=busy>0||!lastBatch?.more}
}
function render(item){const card=document.createElement('article');card.className='card';card.dataset.decision=item.decision;
 const title=document.createElement('b');title.textContent=item.label;card.append(title);
 const id=document.createElement('div');id.className='id';id.textContent=item.article_id;card.append(id);
 const img=document.createElement('img');img.className='photo';img.alt=item.label+' '+item.article_id;img.src='image/'+item.article_id;card.append(img);
 const note=document.createElement('textarea');note.placeholder='Motivo opcional; escribe antes de pulsar un botón';note.value=item.reason;
 note.oninput=()=>note.dataset.dirty='1';
 const actions=document.createElement('div');actions.className='actions';
 for(const [value,label] of Object.entries(names)){const b=document.createElement('button');b.textContent=label;b.dataset.value=value;
 b.setAttribute('aria-pressed',value===item.decision);b.onclick=()=>mark(card,item,value,
 (note.dataset.dirty==='1'||value===item.decision)?note.value:'');actions.append(b)}
 card.append(actions,note);
 const save=document.createElement('button');save.textContent='Guardar nota';save.style.fontSize='12px';
 save.onclick=()=>item.decision?mark(card,item,item.decision,note.value):error(Error('Elige primero Conservar, Excluir o Duda.'));card.append(save);
 const status=document.createElement('div');status.className='status';status.setAttribute('aria-live','polite');
 status.textContent=item.decision?'Guardado · '+names[item.decision]:'Pendiente';card.append(status);
 const auto=document.createElement('div');auto.className='auto';auto.textContent='CLIP original: '+item.automatic+' · puede equivocarse';card.append(auto);
 return card}
async function load(reset=false){if(busy)return;if(reset)cursor='';el('error').textContent='';el('next').disabled=true;
 try{const params=new URLSearchParams({split:el('split').value,label:el('label').value,mode:el('mode').value,after:cursor});
 const data=await api('batch?'+params);lastBatch=data;nextCursor=data.after;
 if(el('label').options.length===1){for(const name of data.labels){const o=document.createElement('option');o.value=name;o.textContent=name;el('label').append(o)}}
 el('grid').replaceChildren(...data.cards.map(render));
 el('info').textContent=`${data.cards.length} fotos en este lote · ${data.remaining} desde esta posición · ${data.saved} decisiones en el registro pendiente. Dataset: ${data.run}`;
 if(!data.cards.length)el('info').textContent+=' · No quedan fotos con estos filtros.';
 el('next').disabled=!data.more;
 }catch(e){error(e)}}
for(const id of ['split','label','mode'])el(id).onchange=()=>load(true);
el('reset').onclick=()=>load(true);
el('next').onclick=()=>{cursor=nextCursor;load()};load();
</script></html>'''


def start_review(root=None, open_browser=True):
    root = Path(root) if root else Path(__file__).resolve().parents[1]
    store = ReviewStore(root)
    token = secrets.token_urlsafe(32)

    class Handler(BaseHTTPRequestHandler):
        def log_message(self, *args):
            pass

        def reply(self, code, data, content_type):
            self.send_response(code)
            self.send_header('Content-Type', content_type)
            self.send_header('Cache-Control', 'no-store')
            self.send_header('X-Content-Type-Options', 'nosniff')
            self.send_header('Content-Length', str(len(data)))
            self.end_headers()
            self.wfile.write(data)

        def route(self):
            expected = f'127.0.0.1:{self.server.server_port}'
            if self.headers.get('Host') != expected:
                raise ValueError('Acceso local no válido')
            parsed = urlparse(self.path)
            prefix = f'/{token}/'
            if not parsed.path.startswith(prefix):
                raise ValueError('Dirección de galería no válida')
            return parsed.path[len(prefix):], parse_qs(parsed.query)

        def do_GET(self):
            try:
                path, query = self.route()
                if path == '':
                    self.reply(200, PAGE.encode('utf-8'), 'text/html; charset=utf-8')
                elif path == 'batch':
                    args = {key: query.get(key, [default])[0] for key, default in
                            [('split', 'train'), ('label', ''), ('mode', 'clip_uncertain'), ('after', '')]}
                    self.reply(200, json.dumps(store.batch(**args)).encode(), 'application/json')
                elif path.startswith('image/'):
                    self.reply(200, store.image(path.removeprefix('image/')), 'image/jpeg')
                else:
                    self.reply(404, b'No encontrado', 'text/plain')
            except (ValueError, KeyError, OSError) as error:
                self.reply(400, str(error).encode('utf-8'), 'text/plain; charset=utf-8')

        def do_POST(self):
            try:
                path, _ = self.route()
                origin = self.headers.get('Origin')
                if origin and origin != f'http://127.0.0.1:{self.server.server_port}':
                    raise ValueError('Origen no válido')
                if path != 'decision':
                    raise ValueError('Acción no válida')
                size = int(self.headers.get('Content-Length', '0'))
                if not 0 < size <= 8192:
                    raise ValueError('Decisión demasiado grande o vacía')
                data = json.loads(self.rfile.read(size))
                result = store.save(str(data['article_id']), data['decision'], str(data.get('reason', '')))
                self.reply(200, json.dumps(result).encode(), 'application/json')
            except (ValueError, KeyError, OSError, TypeError) as error:
                self.reply(400, str(error).encode('utf-8'), 'text/plain; charset=utf-8')

    server = ThreadingHTTPServer(('127.0.0.1', 0), Handler)
    server.daemon_threads = True
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    url = f'http://127.0.0.1:{server.server_port}/{token}/'
    print('Galería local preparada. Cada decisión se guarda automáticamente en:')
    print(store.path)
    print(url)
    if open_browser:
        webbrowser.open(url)
    return server, url


if __name__ == '__main__':
    server, _ = start_review()
    try:
        threading.Event().wait()
    except KeyboardInterrupt:
        server.shutdown()
        server.server_close()
