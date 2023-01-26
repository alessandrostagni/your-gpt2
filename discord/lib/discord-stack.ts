import * as cdk from 'aws-cdk-lib';
import { Construct } from 'constructs';
import * as ecsPatterns from "aws-cdk-lib/aws-ecs-patterns";
import * as ecs from "aws-cdk-lib/aws-ecs"
import * as ec2 from "aws-cdk-lib/aws-ec2"
import { DockerImageAsset, NetworkMode } from 'aws-cdk-lib/aws-ecr-assets';

export class DiscordStack extends cdk.Stack {
  constructor(scope: Construct, id: string, props?: cdk.StackProps) {
    super(scope, id, props);

    const cluster = new ecs.Cluster(this, "DiscordCluster", {
      clusterName: "DiscordCluster"
    })

    const asset = new DockerImageAsset(this, 'DiscordBotDocker', {
      directory: './lib/bot_code',
      networkMode: NetworkMode.HOST,
    })

    const taskDef = new ecs.FargateTaskDefinition(this, "DiscordBotTask", {
      memoryLimitMiB: 512,
      cpu: 256,
    })
    
    taskDef.addContainer("DiscordBotContainer", {
      image: ecs.ContainerImage.fromDockerImageAsset(asset),
      logging: new ecs.AwsLogDriver({
        streamPrefix: "discordbot",
      })
    })

    new ecs.FargateService(this, "DiscordBotService", {
      cluster,
      taskDefinition: taskDef
    });
  }
}
