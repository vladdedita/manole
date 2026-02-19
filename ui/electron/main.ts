import { app, BrowserWindow, dialog, ipcMain, nativeImage, shell } from 'electron'
import { join, resolve, normalize } from 'path'
import { existsSync, lstatSync } from 'fs'
import { execFileSync } from 'child_process'
import { electronApp, optimizer, is } from '@electron-toolkit/utils'
import { PythonBridge } from './python'
import { ModelSetupManager } from './setup'

let mainWindow: BrowserWindow | null = null
const python = new PythonBridge()

const ALLOWED_METHODS = new Set([
  'query', 'init', 'reindex', 'remove_directory',
  'check_models', 'download_models', 'getFileGraph',
  'shutdown', 'ping', 'toggle_debug', 'list_indexes',
])

// Directories the user has explicitly opened via the folder picker
const indexedDirectories = new Set<string>()

const BLOCKED_EXTENSIONS = new Set([
  '.app', '.exe', '.sh', '.command', '.bat', '.cmd',
  '.scr', '.msi', '.dmg', '.pkg', '.deb', '.rpm',
])

function isPathWithinDirectory(filePath: string, directory: string): boolean {
  const resolved = resolve(normalize(filePath))
  const dir = resolve(normalize(directory))
  return resolved.startsWith(dir + '/')
}

function createWindow(): void {
  const iconPath = join(__dirname, '../../build/icon.png')
  mainWindow = new BrowserWindow({
    width: 900,
    height: 700,
    minWidth: 600,
    minHeight: 500,
    show: false,
    icon: iconPath,
    titleBarStyle: 'hiddenInset',
    backgroundColor: '#141210',
    webPreferences: {
      preload: join(__dirname, '../preload/index.js'),
      sandbox: true
    }
  })

  mainWindow.on('ready-to-show', () => {
    mainWindow?.show()
  })

  mainWindow.webContents.setWindowOpenHandler((details) => {
    try {
      const url = new URL(details.url)
      if (url.protocol === 'https:' || url.protocol === 'http:') {
        shell.openExternal(details.url)
      }
    } catch {
      // Invalid URL — ignore
    }
    return { action: 'deny' }
  })

  if (is.dev && process.env['ELECTRON_RENDERER_URL']) {
    mainWindow.loadURL(process.env['ELECTRON_RENDERER_URL'])
  } else {
    mainWindow.loadFile(join(__dirname, '../renderer/index.html'))
  }
}

app.whenReady().then(async () => {
  electronApp.setAppUserModelId('com.manole')
  app.setName('Manole')

  const iconPath = join(__dirname, '../../build/icon.png')
  if (process.platform === 'darwin' && app.dock) {
    app.dock.setIcon(nativeImage.createFromPath(iconPath))
  }

  app.on('browser-window-created', (_, window) => {
    optimizer.watchWindowShortcuts(window)
  })
  createWindow()

  const setup = new ModelSetupManager(python, mainWindow!)
  const modelsDir = setup.getModelsDir()

  python.spawn((response) => {
    mainWindow?.webContents.send('python:message', response)
  }, { MANOLE_MODELS_DIR: modelsDir })

  // Wait for renderer to finish loading before sending setup messages,
  // otherwise they arrive before React mounts its IPC listener
  await new Promise<void>((resolve) => {
    if (mainWindow!.webContents.isLoading()) {
      mainWindow!.webContents.once('did-finish-load', () => resolve())
    } else {
      resolve()
    }
  })

  await setup.checkAndDownload()

  ipcMain.handle('python:send', async (_event, method: string, params?: Record<string, unknown>) => {
    if (!ALLOWED_METHODS.has(method)) {
      throw new Error(`Disallowed method: ${method}`)
    }
    return python.send(method, params ?? {})
  })

  ipcMain.handle('open-file', async (_event, filePath: string) => {
    if (typeof filePath !== 'string' || !filePath) {
      return 'Invalid file path'
    }
    const resolved = resolve(normalize(filePath))

    // Block dangerous file extensions
    const ext = resolved.slice(resolved.lastIndexOf('.')).toLowerCase()
    if (BLOCKED_EXTENSIONS.has(ext)) {
      return 'Blocked file type'
    }

    // Must not be a symlink
    try {
      const stat = lstatSync(resolved)
      if (stat.isSymbolicLink()) {
        return 'Symlinks not allowed'
      }
    } catch {
      return 'File not found'
    }

    // Must be within an indexed directory
    let allowed = false
    for (const dir of indexedDirectories) {
      if (isPathWithinDirectory(resolved, dir)) {
        allowed = true
        break
      }
    }
    if (!allowed) {
      return 'File not in indexed directory'
    }

    return shell.openPath(resolved)
  })

  ipcMain.handle('get-app-metrics', () => {
    // Sum memory from all Electron processes (main, renderer, GPU, etc.)
    const metrics = app.getAppMetrics()
    let electronMemBytes = 0
    let electronCpuPercent = 0
    for (const m of metrics) {
      electronMemBytes += (m.memory.workingSetSize ?? 0) * 1024 // KB → bytes
      electronCpuPercent += m.cpu.percentCPUUsage ?? 0
    }

    // Get Python child process memory via ps (macOS/Linux)
    let pythonMemBytes = 0
    let pythonCpuPercent = 0
    const pid = python.pid
    if (pid) {
      try {
        const out = execFileSync('ps', ['-o', 'rss=,pcpu=', '-p', String(pid)], { encoding: 'utf8' }).trim()
        const parts = out.split(/\s+/)
        if (parts.length >= 2) {
          pythonMemBytes = parseInt(parts[0], 10) * 1024 // KB → bytes
          pythonCpuPercent = parseFloat(parts[1]) || 0
        }
      } catch {
        // Process may have exited
      }
    }

    return {
      memoryBytes: electronMemBytes + pythonMemBytes,
      cpuPercent: Math.round(electronCpuPercent + pythonCpuPercent),
    }
  })

  ipcMain.handle('dialog:openDirectory', async () => {
    const result = await dialog.showOpenDialog(mainWindow!, {
      properties: ['openDirectory'],
    })
    if (result.canceled) return null
    const dirPath = result.filePaths[0]
    indexedDirectories.add(resolve(normalize(dirPath)))
    return dirPath
  })

  app.on('activate', () => {
    if (BrowserWindow.getAllWindows().length === 0) createWindow()
  })
})

app.on('before-quit', () => {
  python.kill()
})

app.on('window-all-closed', () => {
  if (process.platform !== 'darwin') {
    app.quit()
  }
})
