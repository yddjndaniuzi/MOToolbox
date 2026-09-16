import AppKit
import WebKit

final class AppDelegate: NSObject, NSApplicationDelegate {
    private var window: NSWindow!
    private var webView: WKWebView!
    private var backend: BackendController!
    private let serverURL = URL(string: "http://127.0.0.1:5058")!

    func applicationDidFinishLaunching(_ notification: Notification) {
        backend = BackendController(port: 5058)
        createMenu()
        createWindow()
        startBackendAndLoad()
        NSApp.activate(ignoringOtherApps: true)
    }

    func applicationShouldTerminateAfterLastWindowClosed(_ sender: NSApplication) -> Bool {
        true
    }

    func applicationWillTerminate(_ notification: Notification) {
        backend.stop()
    }

    @objc private func reloadPage() {
        webView.reload()
    }

    @objc private func goBack() {
        if webView.canGoBack {
            webView.goBack()
        }
    }

    @objc private func goForward() {
        if webView.canGoForward {
            webView.goForward()
        }
    }

    @objc private func openInBrowser() {
        NSWorkspace.shared.open(serverURL)
    }

    private func createMenu() {
        let mainMenu = NSMenu()
        let appMenuItem = NSMenuItem()
        let appMenu = NSMenu()
        appMenu.addItem(
            NSMenuItem(
                title: "Quit MOtoolbox",
                action: #selector(NSApplication.terminate(_:)),
                keyEquivalent: "q"
            )
        )
        appMenuItem.submenu = appMenu
        mainMenu.addItem(appMenuItem)

        let fileMenuItem = NSMenuItem()
        let fileMenu = NSMenu(title: "File")
        fileMenu.addItem(
            NSMenuItem(title: "Close Window", action: #selector(NSWindow.performClose(_:)), keyEquivalent: "w")
        )
        fileMenuItem.submenu = fileMenu
        mainMenu.addItem(fileMenuItem)

        // 没有 Edit 菜单时，WKWebView 内的 Cmd+C/V/X/A/Z 等键等价项不会进入响应链。
        let editMenuItem = NSMenuItem()
        let editMenu = NSMenu(title: "Edit")
        editMenu.addItem(NSMenuItem(title: "Undo", action: Selector(("undo:")), keyEquivalent: "z"))
        let redoItem = NSMenuItem(title: "Redo", action: Selector(("redo:")), keyEquivalent: "Z")
        editMenu.addItem(redoItem)
        editMenu.addItem(NSMenuItem.separator())
        editMenu.addItem(NSMenuItem(title: "Cut", action: #selector(NSText.cut(_:)), keyEquivalent: "x"))
        editMenu.addItem(NSMenuItem(title: "Copy", action: #selector(NSText.copy(_:)), keyEquivalent: "c"))
        editMenu.addItem(NSMenuItem(title: "Paste", action: #selector(NSText.paste(_:)), keyEquivalent: "v"))
        editMenu.addItem(NSMenuItem(title: "Select All", action: #selector(NSText.selectAll(_:)), keyEquivalent: "a"))
        editMenuItem.submenu = editMenu
        mainMenu.addItem(editMenuItem)

        let viewMenuItem = NSMenuItem()
        let viewMenu = NSMenu(title: "View")
        viewMenu.addItem(NSMenuItem(title: "Reload", action: #selector(reloadPage), keyEquivalent: "r"))
        viewMenu.addItem(NSMenuItem(title: "Back", action: #selector(goBack), keyEquivalent: "["))
        viewMenu.addItem(NSMenuItem(title: "Forward", action: #selector(goForward), keyEquivalent: "]"))
        viewMenu.addItem(NSMenuItem(title: "Open in Browser", action: #selector(openInBrowser), keyEquivalent: "o"))
        viewMenuItem.submenu = viewMenu
        mainMenu.addItem(viewMenuItem)
        NSApp.mainMenu = mainMenu
    }

    private func createWindow() {
        let configuration = WKWebViewConfiguration()
        configuration.defaultWebpagePreferences.allowsContentJavaScript = true
        webView = WKWebView(frame: .zero, configuration: configuration)
        webView.allowsBackForwardNavigationGestures = true

        window = NSWindow(
            contentRect: NSRect(x: 0, y: 0, width: 1280, height: 860),
            styleMask: [.titled, .closable, .miniaturizable, .resizable, .fullSizeContentView],
            backing: .buffered,
            defer: false
        )
        window.title = "MOtoolbox"
        window.minSize = NSSize(width: 960, height: 640)
        window.center()
        window.contentView = webView
        window.toolbar = makeToolbar()
        window.makeKeyAndOrderFront(nil)
    }

    private func makeToolbar() -> NSToolbar {
        let toolbar = NSToolbar(identifier: "MOtoolboxToolbar")
        toolbar.displayMode = .iconOnly
        toolbar.delegate = self
        return toolbar
    }

    private func startBackendAndLoad() {
        showLoadingPage()
        backend.startIfNeeded { [weak self] result in
            DispatchQueue.main.async {
                guard let self else { return }
                switch result {
                case .success:
                    self.webView.load(URLRequest(url: self.serverURL))
                case .failure(let error):
                    self.showErrorPage(error)
                }
            }
        }
    }

    private func showLoadingPage() {
        webView.loadHTMLString(
            """
            <!doctype html>
            <html>
            <head>
              <meta charset="utf-8">
              <style>
                body { margin: 0; height: 100vh; display: grid; place-items: center; font: 15px -apple-system, BlinkMacSystemFont, sans-serif; color: #1f2937; background: #f8fafc; }
                main { text-align: center; }
                h1 { margin: 0 0 8px; font-size: 18px; font-weight: 650; }
                p { margin: 0; color: #64748b; }
              </style>
            </head>
            <body><main><h1>Starting MOtoolbox</h1><p>Preparing the local workspace...</p></main></body>
            </html>
            """,
            baseURL: nil
        )
    }

    private func showErrorPage(_ error: Error) {
        let message = String(describing: error)
            .replacingOccurrences(of: "&", with: "&amp;")
            .replacingOccurrences(of: "<", with: "&lt;")
            .replacingOccurrences(of: ">", with: "&gt;")
        webView.loadHTMLString(
            """
            <!doctype html>
            <html>
            <head>
              <meta charset="utf-8">
              <style>
                body { margin: 0; height: 100vh; display: grid; place-items: center; font: 15px -apple-system, BlinkMacSystemFont, sans-serif; color: #111827; background: #fff7ed; }
                main { width: min(680px, calc(100vw - 64px)); }
                h1 { margin: 0 0 12px; font-size: 20px; }
                pre { white-space: pre-wrap; background: #ffffff; border: 1px solid #fed7aa; border-radius: 8px; padding: 14px; color: #7c2d12; }
              </style>
            </head>
            <body><main><h1>MOtoolbox could not start</h1><pre>\(message)</pre></main></body>
            </html>
            """,
            baseURL: nil
        )
    }
}

extension AppDelegate: NSToolbarDelegate {
    func toolbarAllowedItemIdentifiers(_ toolbar: NSToolbar) -> [NSToolbarItem.Identifier] {
        [.back, .forward, .reload, .openExternal, .flexibleSpace]
    }

    func toolbarDefaultItemIdentifiers(_ toolbar: NSToolbar) -> [NSToolbarItem.Identifier] {
        [.back, .forward, .reload, .flexibleSpace, .openExternal]
    }

    func toolbar(
        _ toolbar: NSToolbar,
        itemForItemIdentifier itemIdentifier: NSToolbarItem.Identifier,
        willBeInsertedIntoToolbar flag: Bool
    ) -> NSToolbarItem? {
        let item = NSToolbarItem(itemIdentifier: itemIdentifier)
        switch itemIdentifier {
        case .back:
            item.label = "Back"
            item.paletteLabel = "Back"
            item.toolTip = "Go back"
            item.image = NSImage(systemSymbolName: "chevron.left", accessibilityDescription: "Back")
            item.action = #selector(goBack)
        case .forward:
            item.label = "Forward"
            item.paletteLabel = "Forward"
            item.toolTip = "Go forward"
            item.image = NSImage(systemSymbolName: "chevron.right", accessibilityDescription: "Forward")
            item.action = #selector(goForward)
        case .reload:
            item.label = "Reload"
            item.paletteLabel = "Reload"
            item.toolTip = "Reload"
            item.image = NSImage(systemSymbolName: "arrow.clockwise", accessibilityDescription: "Reload")
            item.action = #selector(reloadPage)
        case .openExternal:
            item.label = "Browser"
            item.paletteLabel = "Open in Browser"
            item.toolTip = "Open in browser"
            item.image = NSImage(systemSymbolName: "safari", accessibilityDescription: "Open in browser")
            item.action = #selector(openInBrowser)
        default:
            return nil
        }
        item.target = self
        return item
    }
}

private extension NSToolbarItem.Identifier {
    static let back = NSToolbarItem.Identifier("Back")
    static let forward = NSToolbarItem.Identifier("Forward")
    static let reload = NSToolbarItem.Identifier("Reload")
    static let openExternal = NSToolbarItem.Identifier("OpenExternal")
}
