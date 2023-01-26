import * as cdk from 'aws-cdk-lib';
import { Construct } from 'constructs';
import * as ecsPatterns from "aws-cdk-lib/aws-ecs-patterns";
import * as ecs from "aws-cdk-lib/aws-ecs"
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

    new ecsPatterns.ApplicationLoadBalancedEc2Service(this, 'Service', {
      cluster,
      memoryLimitMiB: 1024,
      taskImageOptions: {
        image: ecs.ContainerImage.fromDockerImageAsset(asset)
      },
      desiredCount: 1,
    });
  }
}
