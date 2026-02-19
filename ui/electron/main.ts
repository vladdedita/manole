import { app, BrowserWindow, dialog, ipcMain, shell } from 'electron'
import { join } from 'path'
import { execFileSync } from 'child_process'
import { electronApp, optimizer, is } from '@electron-toolkit/utils'
import { PythonBridge } from './python'
import { ModelSetupManager } from './setup'

let mainWindow: BrowserWindow | null = null
const python = new PythonBridge()

function createWindow(): void {
  mainWindow = new BrowserWindow({
    width: 900,
    height: 700,
    minWidth: 600,
    minHeight: 500,
    show: false,
    titleBarStyle: 'hiddenInset',
    backgroundColor: '#141210',
    webPreferences: {
      preload: join(__dirname, '../preload/index.js'),
      sandbox: false
    }
  })

  mainWindow.on('ready-to-show', () => {
    mainWindow?.show()
  })

  mainWindow.webContents.setWindowOpenHandler((details) => {
    shell.openExternal(details.url)
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
  app.on('browser-window-created', (_, window) => {
    optimizer.watchWindowShortcuts(window)
  })
  createWindow()

  const setup = new ModelSetupManager(python, mainWindow!)
  const modelsDir = setup.getModelsDir()

  python.spawn((response) => {
    mainWindow?.webContents.send('python:message', response)
  }, { MANOLE_MODELS_DIR: modelsDir })

  await setup.checkAndDownload()

  ipcMain.handle('python:send', async (_event, method: string, params?: Record<string, unknown>) => {
    return python.send(method, params ?? {})
  })

  ipcMain.handle('open-file', async (_event, filePath: string) => {
    return shell.openPath(filePath)
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
    return result.filePaths[0]
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
