# -*- mode: python ; coding: utf-8 -*-
a = Analysis(
    ['server.py'],
    pathex=['.'],
    datas=[],
    hiddenimports=[
        'leann',
        'llama_cpp',
        'docling',
        'models',
        'agent',
        'searcher',
        'tools',
        'toolbox',
        'router',
        'rewriter',
        'parser',
        'file_reader',
        'chat',
    ],
    hookspath=[],
    runtime_hooks=[],
    excludes=['tkinter', 'matplotlib', 'IPython', 'notebook'],
)

pyz = PYZ(a.pure, a.zipped_data)

exe = EXE(
    pyz,
    a.scripts,
    a.binaries,
    a.datas,
    name='manole-server',
    strip=False,
    upx=True,
    console=True,
)
