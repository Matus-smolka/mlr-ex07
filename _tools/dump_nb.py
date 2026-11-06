import json, io, sys

nb_path, out_path = sys.argv[1], sys.argv[2]
nb = json.load(open(nb_path, encoding='utf-8'))
out = io.open(out_path, 'w', encoding='utf-8')
out.write('cells: %d\n' % len(nb['cells']))
for i, c in enumerate(nb['cells']):
    src = ''.join(c['source'])
    out.write('\n===== CELL %d [%s] chars=%d =====\n' % (i, c['cell_type'], len(src)))
    out.write(src)
    if c.get('outputs'):
        out.write('\n--- OUTPUTS (%d) ---\n' % len(c['outputs']))
out.close()
print('ok')
