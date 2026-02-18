import { contextBridge, ipcRenderer } from 'electron'

contextBridge.exposeInMainWorld('api', {
  send: (method: string, params?: Record<string, unknown>) =>
    ipcRenderer.invoke('python:send', method, params),
  onMessage: (callback: (response: unknown) => void) => {
    const listener = (_event: unknown, response: unknown) => callback(response)
    ipcRenderer.on('python:message', listener)
    return () => {
      ipcRenderer.removeListener('python:message', listener)
    }
  },
  selectDirectory: () => ipcRenderer.invoke('dialog:openDirectory'),
  openFile: (filePath: string) => ipcRenderer.invoke('open-file', filePath),
  getAppMetrics: () => ipcRenderer.invoke('get-app-metrics') as Promise<{ memoryBytes: number; cpuPercent: number }>
})
