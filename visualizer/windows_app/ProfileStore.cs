using System;
using System.Collections.Generic;
using System.IO;
using System.Linq;
using System.Text;
using System.Web.Script.Serialization;

namespace SnakeRobotVisualizer
{
    internal sealed class ProfileStore
    {
        private readonly string baseDirectory;
        private readonly string profilesDirectory;
        private readonly string statePath;
        private readonly JavaScriptSerializer json = new JavaScriptSerializer();

        public ProfileStore(string executableDirectory)
        {
            baseDirectory = executableDirectory;
            profilesDirectory = Path.Combine(baseDirectory, "profiles");
            statePath = Path.Combine(baseDirectory, "app-state.json");
            Directory.CreateDirectory(profilesDirectory);
        }

        public string ProfilesDirectory { get { return profilesDirectory; } }

        public void EnsureDefaultProfiles()
        {
            if (Directory.GetFiles(profilesDirectory, "*.json").Length > 0)
            {
                return;
            }
            SaveProfile(RobotProfile.CreateDefaultTwoJoint());
            SaveProfile(RobotProfile.CreateAlternating(28, "28关节模拟档案"));
        }

        public List<RobotProfile> LoadProfiles()
        {
            EnsureDefaultProfiles();
            List<RobotProfile> result = new List<RobotProfile>();
            foreach (string path in Directory.GetFiles(profilesDirectory, "*.json").OrderBy(x => x))
            {
                try
                {
                    RobotProfile profile = json.Deserialize<RobotProfile>(File.ReadAllText(path, Encoding.UTF8));
                    if (profile == null)
                    {
                        continue;
                    }
                    if (profile.Joints == null)
                    {
                        profile.Joints = new List<JointConfig>();
                    }
                    profile.Reindex();
                    result.Add(profile);
                }
                catch
                {
                    // Invalid user profile files are ignored and can be repaired from the UI.
                }
            }
            if (result.Count == 0)
            {
                RobotProfile fallback = RobotProfile.CreateDefaultTwoJoint();
                SaveProfile(fallback);
                result.Add(fallback);
            }
            return result;
        }

        public RobotProfile LoadProfile(string name)
        {
            return LoadProfiles().FirstOrDefault(x => x.Name == name);
        }

        public void SaveProfile(RobotProfile profile)
        {
            if (profile == null)
            {
                throw new ArgumentNullException("profile");
            }
            profile.Reindex();
            List<string> errors = profile.Validate();
            if (errors.Count > 0)
            {
                throw new InvalidOperationException(String.Join("\r\n", errors.ToArray()));
            }
            string path = Path.Combine(profilesDirectory, SafeFileName(profile.Name) + ".json");
            File.WriteAllText(path, json.Serialize(profile), new UTF8Encoding(false));
        }

        public RobotProfile Duplicate(RobotProfile source, string newName)
        {
            RobotProfile copy = source.Clone();
            copy.Name = newName;
            SaveProfile(copy);
            return copy;
        }

        public ApplicationState LoadApplicationState()
        {
            try
            {
                if (File.Exists(statePath))
                {
                    ApplicationState state =
                        json.Deserialize<ApplicationState>(File.ReadAllText(statePath, Encoding.UTF8));
                    if (state != null)
                    {
                        if (state.Display == null)
                        {
                            state.Display = new DisplaySettings();
                        }
                        return state;
                    }
                }
            }
            catch
            {
            }
            return new ApplicationState();
        }

        public void SaveApplicationState(ApplicationState state)
        {
            File.WriteAllText(statePath, json.Serialize(state), new UTF8Encoding(false));
        }

        private static string SafeFileName(string name)
        {
            string value = String.IsNullOrWhiteSpace(name) ? "未命名档案" : name.Trim();
            foreach (char invalid in Path.GetInvalidFileNameChars())
            {
                value = value.Replace(invalid, '_');
            }
            return value;
        }
    }
}
