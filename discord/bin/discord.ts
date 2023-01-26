#!/usr/bin/env node
import 'source-map-support/register';
import * as cdk from 'aws-cdk-lib';
import { DiscordStack } from '../lib/discord-stack';

const app = new cdk.App();
new DiscordStack(app, 'DiscordStack', {
  env: { 
    account: "762998815470", 
    region: process.env.CDK_DEFAULT_REGION 
  }
});