import Foundation

final class BackendController {
    enum BackendError: Error, CustomStringConvertible {
        case repositoryNotFound
        case startScriptNotFound(URL)
        case bundledBackendNotFound(URL)
        case startupTimedOut

        var description: String {
            switch self {
            case .repositoryNotFound:
                return "Could not find the MOtoolbox repository. Set MOTOOLBOX_REPO_ROOT when launching the app."
            case .startScriptNotFound(let url):
                return "Could not find start_web.sh at \(url.path)."
            case .bundledBackendNotFound(let url):
                return "Could not find bundled backend executable at \(url.path)."
            case .startupTimedOut:
                return "The local web server did not become ready in time."
            }
        }
    }

    private let host = "127.0.0.1"
    private let port: Int
    private var process: Process?
    private var ownsProcess = false

    init(port: Int) {
        self.port = port
    }

    func startIfNeeded(completion: @escaping (Result<Void, Error>) -> Void) {
        healthCheck { [weak self] healthy in
            guard let self else { return }
            if healthy {
                completion(.success(()))
                return
            }

            do {
                try self.start()
            } catch {
                completion(.failure(error))
                return
            }

            self.waitUntilReady(deadline: Date().addingTimeInterval(45), completion: completion)
        }
    }

    func stop() {
        guard ownsProcess, let process, process.isRunning else {
            return
        }
        process.terminate()
    }

    private func start() throws {
        if let bundledBackend = bundledBackendExecutable() {
            try startBundledBackend(bundledBackend)
            return
        }

        try startDevelopmentBackend()
    }

    private func startBundledBackend(_ executable: URL) throws {
        guard FileManager.default.isExecutableFile(atPath: executable.path) else {
            throw BackendError.bundledBackendNotFound(executable)
        }
        let process = Process()
        process.executableURL = executable
        process.arguments = []
        process.currentDirectoryURL = executable.deletingLastPathComponent()

        var environment = ProcessInfo.processInfo.environment
        environment["MOTOOLBOX_PORT"] = String(port)
        environment["MOTOOLBOX_NO_BROWSER"] = "1"
        environment["PYTHONUNBUFFERED"] = "1"
        process.environment = environment

        try attachLogsAndRun(process)
    }

    private func startDevelopmentBackend() throws {
        let repository = try repositoryRoot()
        let script = repository.appendingPathComponent("start_web.sh")
        guard FileManager.default.isExecutableFile(atPath: script.path) else {
            throw BackendError.startScriptNotFound(script)
        }

        let process = Process()
        process.executableURL = URL(fileURLWithPath: "/bin/zsh")
        process.arguments = ["-lc", "./start_web.sh"]
        process.currentDirectoryURL = repository

        var environment = ProcessInfo.processInfo.environment
        environment["MOTOOLBOX_PORT"] = String(port)
        environment["PYTHONUNBUFFERED"] = "1"
        process.environment = environment

        try attachLogsAndRun(process)
    }

    private func attachLogsAndRun(_ process: Process) throws {
        let logs = URL(fileURLWithPath: "/tmp")
        let stdout = logs.appendingPathComponent("motoolbox-shell-web.log")
        let stderr = logs.appendingPathComponent("motoolbox-shell-web.err")
        process.standardOutput = try FileHandle(forWritingTo: stdout, createIfNeeded: true)
        process.standardError = try FileHandle(forWritingTo: stderr, createIfNeeded: true)

        try process.run()
        self.process = process
        ownsProcess = true
    }

    private func bundledBackendExecutable() -> URL? {
        guard let resources = Bundle.main.resourceURL else {
            return nil
        }
        let executable = resources
            .appendingPathComponent("MOtoolboxBackend.app")
            .appendingPathComponent("Contents")
            .appendingPathComponent("MacOS")
            .appendingPathComponent("MOtoolboxBackend")
        return FileManager.default.fileExists(atPath: executable.path) ? executable : nil
    }

    private func waitUntilReady(deadline: Date, completion: @escaping (Result<Void, Error>) -> Void) {
        healthCheck { [weak self] healthy in
            guard let self else { return }
            if healthy {
                completion(.success(()))
                return
            }
            if Date() >= deadline {
                completion(.failure(BackendError.startupTimedOut))
                return
            }
            DispatchQueue.global().asyncAfter(deadline: .now() + 0.35) {
                self.waitUntilReady(deadline: deadline, completion: completion)
            }
        }
    }

    private func healthCheck(completion: @escaping (Bool) -> Void) {
        let url = URL(string: "http://\(host):\(port)/healthz")!
        var request = URLRequest(url: url)
        request.timeoutInterval = 0.6
        URLSession.shared.dataTask(with: request) { _, response, _ in
            let status = (response as? HTTPURLResponse)?.statusCode
            completion(status == 200)
        }.resume()
    }

    private func repositoryRoot() throws -> URL {
        let fileManager = FileManager.default
        if let configured = ProcessInfo.processInfo.environment["MOTOOLBOX_REPO_ROOT"], !configured.isEmpty {
            let url = URL(fileURLWithPath: configured).standardizedFileURL
            if fileManager.fileExists(atPath: url.appendingPathComponent("start_web.sh").path) {
                return url
            }
        }

        var url = URL(fileURLWithPath: #filePath)
        for _ in 0..<8 {
            url.deleteLastPathComponent()
            if fileManager.fileExists(atPath: url.appendingPathComponent("start_web.sh").path) {
                return url
            }
        }

        throw BackendError.repositoryNotFound
    }
}

private extension FileHandle {
    convenience init(forWritingTo url: URL, createIfNeeded: Bool) throws {
        if createIfNeeded && !FileManager.default.fileExists(atPath: url.path) {
            FileManager.default.createFile(atPath: url.path, contents: nil)
        }
        try self.init(forWritingTo: url)
        try seekToEnd()
    }
}
