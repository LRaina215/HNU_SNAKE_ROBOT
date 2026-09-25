using System;
using System.Collections.Generic;
using System.IO;
using System.Net;
using System.Net.Sockets;
using System.Text;
using System.Threading;
using System.Web.Script.Serialization;

namespace SnakeRobotVisualizer
{
    internal sealed class LocalControlServer : IDisposable
    {
        private readonly int port;
        private readonly Func<Dictionary<string, object>, Dictionary<string, object>> handler;
        private readonly Action<string> disconnected;
        private readonly object clientsSync = new object();
        private readonly List<TcpClient> clients = new List<TcpClient>();
        private TcpListener listener;
        private Thread thread;
        private volatile bool stopping;

        public LocalControlServer(int portNumber,
            Func<Dictionary<string, object>, Dictionary<string, object>> commandHandler,
            Action<string> onDisconnected = null)
        { port = portNumber; handler = commandHandler; disconnected = onDisconnected; }

        public bool Start()
        {
            try
            {
                listener = new TcpListener(IPAddress.Loopback, port);
                listener.Start();
                stopping = false;
                thread = new Thread(AcceptLoop);
                thread.Name = "Local Python control bridge";
                thread.IsBackground = true;
                thread.Start();
                return true;
            }
            catch { return false; }
        }

        private void AcceptLoop()
        {
            while (!stopping)
            {
                try
                {
                    TcpClient client = listener.AcceptTcpClient();
                    lock (clientsSync)
                    {
                        if (stopping || clients.Count >= 32) { client.Close(); continue; }
                        clients.Add(client);
                    }
                    ThreadPool.QueueUserWorkItem(delegate { HandleClient(client); });
                }
                catch { if (!stopping) Thread.Sleep(50); }
            }
        }

        private void HandleClient(TcpClient client)
        {
            string connectionId = Guid.NewGuid().ToString("N");
            try
            {
                client.ReceiveTimeout = 15000;
                client.SendTimeout = 2000;
                client.NoDelay = true;
                using (client)
                using (NetworkStream stream = client.GetStream())
                using (StreamReader reader = new StreamReader(stream, new UTF8Encoding(false)))
                using (StreamWriter writer = new StreamWriter(stream, new UTF8Encoding(false)))
                {
                    writer.AutoFlush = true;
                    JavaScriptSerializer json = new JavaScriptSerializer();
                    json.MaxJsonLength = 1048576;
                    while (!stopping)
                    {
                        string line = ReadLineLimited(reader);
                        if (line == null) break;
                        Dictionary<string, object> response;
                        try
                        {
                            Dictionary<string, object> request = json.Deserialize<Dictionary<string, object>>(line);
                            if (request == null) throw new ArgumentException("请求必须为 JSON 对象");
                            request["_connection_id"] = connectionId;
                            response = handler(request);
                        }
                        catch (Exception ex)
                        {
                            response = new Dictionary<string, object> { { "ok", false }, { "error", ex.Message } };
                        }
                        writer.WriteLine(json.Serialize(response));
                    }
                }
            }
            catch (IOException) { }
            catch (SocketException) { }
            catch (ObjectDisposedException) { }
            finally
            {
                client.Close();
                lock (clientsSync) { clients.Remove(client); }
                if (disconnected != null) disconnected(connectionId);
            }
        }

        private static string ReadLineLimited(StreamReader reader)
        {
            StringBuilder line = new StringBuilder();
            for (int count = 0; count <= 1048576; count++)
            {
                int c = reader.Read();
                if (c < 0) return null; // 不执行缺少换行的半包。
                if (c == '\n') return line.ToString();
                line.Append((char)c);
            }
            throw new IOException("JSON 帧超过 1 MiB");
        }

        public void Stop()
        {
            stopping = true;
            if (listener != null) { try { listener.Stop(); } catch { } }
            TcpClient[] open;
            lock (clientsSync) { open = clients.ToArray(); }
            foreach (TcpClient client in open) { try { client.Close(); } catch { } }
            if (thread != null && thread.IsAlive) thread.Join(300);
            thread = null;
            listener = null;
        }
        public void Dispose() { Stop(); }
    }
}
