// Research-only stdin/stdout adapter around the shipped estimator assembly.
// Seeds its last-valid axis from a checkpoint; does not modify the GUI process.
using System;
using System.Collections;
using System.Collections.Generic;
using System.Reflection;
using System.Web.Script.Serialization;

class SegmentedAxisReplay
{
    static BindingFlags Flags = BindingFlags.Instance | BindingFlags.Public | BindingFlags.NonPublic;
    static Type Point, Axis, Estimator, Projection;
    static object MakePoint(object value) {
        var a = (ArrayList)value;
        return Activator.CreateInstance(Point, new object[] { Convert.ToDouble(a[0]), Convert.ToDouble(a[1]), Convert.ToDouble(a[2]) });
    }
    static double[] Vec(object point) {
        return new double[] { (double)Point.GetField("X").GetValue(point), (double)Point.GetField("Y").GetValue(point), (double)Point.GetField("Z").GetValue(point) };
    }
    static void Main(string[] args) {
        var assembly = Assembly.LoadFrom(args[0]);
        Point = assembly.GetType("SnakeRobotVisualizer.Point3D");
        Axis = assembly.GetType("SnakeRobotVisualizer.RollingAxisState");
        Estimator = assembly.GetType("SnakeRobotVisualizer.RollingAxisEstimator");
        Projection = assembly.GetType("SnakeRobotVisualizer.RollingProjection");
        var estimator = Activator.CreateInstance(Estimator, true);
        var serializer = new JavaScriptSerializer();
        object previous = null;
        string line;
        while ((line=Console.ReadLine()) != null) {
            var input = serializer.Deserialize<Dictionary<string,object>>(line);
            if (input.ContainsKey("smoothing"))
                Estimator.GetProperty("Smoothing").SetValue(estimator,Convert.ToDouble(input["smoothing"]),null);
            if (input.ContainsKey("seed")) {
                var seed = (Dictionary<string,object>)input["seed"];
                previous = Activator.CreateInstance(Axis,true);
                foreach (var pair in new string[][] {new[]{"Point","point_mm"},new[]{"Direction","direction"},new[]{"Horizontal","horizontal"},new[]{"Vertical","vertical"}})
                    Axis.GetField(pair[0]).SetValue(previous,MakePoint(seed[pair[1]]));
                Axis.GetField("Valid").SetValue(previous,true);
                Axis.GetField("HasBasis").SetValue(previous,true);
                Estimator.GetField("lastValid",Flags).SetValue(estimator,previous);
            }
            var rows = (ArrayList)input["points_mm"];
            var points = Array.CreateInstance(Point,rows.Count);
            for (int i=0;i<rows.Count;i++) points.SetValue(MakePoint(rows[i]),i);
            var update=Estimator.GetMethod("Update",new Type[]{Point.MakeArrayType(),typeof(bool)});
            var axis=update.Invoke(estimator,new object[]{points,true});
            Projection.GetMethod("SetBasis").Invoke(null,new object[]{axis,previous,false});
            previous=axis;
            var output=new Dictionary<string,object>();
            foreach (var pair in new string[][] {new[]{"Point","point_mm"},new[]{"Direction","direction"},new[]{"Horizontal","horizontal"},new[]{"Vertical","vertical"}})
                output[pair[1]]=Vec(Axis.GetField(pair[0]).GetValue(axis));
            foreach (var pair in new string[][] {new[]{"Confidence","confidence"},new[]{"Valid","valid"},new[]{"Fallback","fallback"},new[]{"Status","status"}})
                output[pair[1]]=Axis.GetField(pair[0]).GetValue(axis);
            Console.WriteLine(serializer.Serialize(output));
        }
    }
}
