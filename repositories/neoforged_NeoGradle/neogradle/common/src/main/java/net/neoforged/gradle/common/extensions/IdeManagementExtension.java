package net.neoforged.gradle.common.extensions;

import net.neoforged.gradle.common.tasks.IdePostSyncExecutionTask;
import net.neoforged.gradle.dsl.common.extensions.subsystems.Conventions;
import net.neoforged.gradle.dsl.common.extensions.subsystems.Subsystems;
import net.neoforged.gradle.dsl.common.extensions.subsystems.conventions.IDE;
import net.neoforged.gradle.dsl.common.extensions.subsystems.conventions.ide.IDEA;
import org.gradle.api.Project;
import org.gradle.api.Task;
import org.gradle.api.plugins.ExtensionAware;
import org.gradle.api.tasks.TaskProvider;
import org.gradle.plugins.ide.eclipse.EclipsePlugin;
import org.gradle.plugins.ide.eclipse.model.EclipseModel;
import org.gradle.plugins.ide.idea.model.IdeaModel;
import org.jetbrains.annotations.NotNull;
import org.jetbrains.gradle.ext.IdeaExtPlugin;
import org.jetbrains.gradle.ext.ProjectSettings;
import org.jetbrains.gradle.ext.TaskTriggersConfig;

import javax.inject.Inject;

/**
 * Defines interactions with the IDE plugins.
 */
public abstract class IdeManagementExtension {

    public static final String IDE_POST_SYNC_TASK_NAME = "idePostSync";

    private final Project project;
    private final Project rootProject;

    @Inject
    public IdeManagementExtension(Project project) {
        this.project = project;
        this.rootProject = project.getRootProject();
        
        project.getPlugins().apply(IdeaExtPlugin.class);
        project.getPlugins().apply(EclipsePlugin.class);
        
        if (project != rootProject) {
            if (!rootProject.getPlugins().hasPlugin(IdeaExtPlugin.class))
                rootProject.getPlugins().apply(IdeaExtPlugin.class);
            
            if (!rootProject.getPlugins().hasPlugin(EclipsePlugin.class))
                rootProject.getPlugins().apply(EclipsePlugin.class);
        }

        // Always pre-create the idePostSync task if IntelliJ is importing the project, since
        // IntelliJ remembers to run the task post-sync even if the import fails. That will cause
        // situations where import errors (i.e. dependency resolution errors) will be masked by
        // the failed idePostSync task, since it was never created in that particular import.
        if (isIdeaImport()) {
            getOrCreateIdeImportTask();
        }
    }

    /**
     * Get whether Gradle is being invoked through IntelliJ IDEA.
     *
     * <p>This can be through a project import, or a task execution.</p>
     *
     * @return whether this is an IntelliJ-based invocation
     */
    public boolean isIdeaImport() {
        return Boolean.getBoolean("idea.active");
    }

    /**
     * Get whether this Gradle invocation is from an Eclipse project import.
     *
     * @return whether an eclipse import is ongoing
     */
    public boolean isEclipseImport() {
        return System.getProperty("eclipse.application") != null;
    }

    /**
     * Indicates if an IDE import in any of the supported IDEs is ongoing.
     *
     * @return {@code true} if an IDE import is ongoing, {@code false} otherwise
     */
    public boolean isIdeImportInProgress() {
        return isIdeaImport() || isEclipseImport();
    }

    /**
     * Configures the current project to run a task after the IDE import is complete.
     *
     * @param taskToRun The task to run
     */
    public void registerTaskToRun(TaskProvider<?> taskToRun) {
        final TaskProvider<? extends Task> idePostSyncTask = getOrCreateIdeImportTask();
        //Configure the idePostSync task to depend on the task to run, causing the past in task to become part of the task-tree that is ran after import.
        idePostSyncTask.configure(task -> task.dependsOn(taskToRun));
    }
    
    @NotNull
    public TaskProvider<? extends Task> getOrCreateIdeImportTask() {
        final TaskProvider<? extends Task> idePostSyncTask;
        //Check for the existence of the idePostSync task, which is created by us as a central entry point for all IDE post-sync tasks
        if (!project.getTasks().getNames().contains(IDE_POST_SYNC_TASK_NAME)) {

            //None found -> Create one.
            idePostSyncTask = project.getTasks().register(IDE_POST_SYNC_TASK_NAME, IdePostSyncExecutionTask.class);

            //Register the task to run after the IDE import is complete
            apply(new IdeImportAction() {
                @Override
                public void idea(Project project, IdeaModel idea, ProjectSettings ideaExtension) {
                    final Conventions conventions = project.getExtensions().getByType(Subsystems.class).getConventions();
                    final IDE ideConventions = conventions.getIde();
                    final IDEA ideaConventions = ideConventions.getIdea();
                    if (!ideaConventions.getShouldUsePostSyncTask().get() &&
                            ideaConventions.getIsEnabled().get())
                        return;

                    //Register the task to run after the IDEA import is complete, via its custom extension.
                    final TaskTriggersConfig taskTriggers = ((ExtensionAware) ideaExtension).getExtensions().getByType(TaskTriggersConfig.class);
                    taskTriggers.afterSync(idePostSyncTask);
                }

                @Override
                public void eclipse(Project project, EclipseModel eclipse) {
                    //Register the task to run after the Eclipse import is complete, via its build-in support.
                    eclipse.synchronizationTasks(idePostSyncTask);
                }
            });
        }
        else {
            //Found -> Use it.
            idePostSyncTask = project.getTasks().named(IDE_POST_SYNC_TASK_NAME);
        }
        return idePostSyncTask;
    }
    
    /**
     * Applies the specified configuration action to configure IDE projects.
     *
     * <p>This does not apply the IDEs' respective plugins, but will perform
     * actions when those plugins are applied.</p>
     *
     * @param toPerform the actions to perform
     */
    public void apply(final IdeImportAction toPerform) {
        onIdea(toPerform);
        onEclipse(toPerform);
        onGradle(toPerform);
    }
    
    /**
     * Applies the specified configuration action to configure idea IDE projects only.
     *
     * <p>This does not apply the idea IDEs' respective plugins, but will perform
     * actions when those plugins are applied.</p>
     *
     * @param toPerform the actions to perform
     */
    public void onIdea(final IdeaIdeImportAction toPerform) {
        //When the IDEA plugin is available, configure it
        project.getPlugins().withType(IdeaExtPlugin.class, plugin -> {
            if (!isIdeaImport()) {
                //No IDEA import even though the plugin is available, so don't configure it.
                return;
            }
            
            //Grab the idea runtime model so we can extend it. -> This is done from the root project, so that the model is available to all subprojects.
            //And so that post sync tasks are only ran once for all subprojects.
            IdeaModel model = project.getExtensions().findByType(IdeaModel.class);
            if (model == null || model.getProject() == null) {
                model = rootProject.getExtensions().findByType(IdeaModel.class);
            }
            
            //If we still don't have a model, throw an exception:
            if (model == null || model.getProject() == null) {
                throw new IllegalStateException("IDEA model is null, even though the IDEA plugin is applied.");
            }
            
            //Locate the project settings on the model and then configure them and then pass them to the action so that it can configure it.
            final ProjectSettings ideaExt = ((ExtensionAware) model.getProject()).getExtensions().getByType(ProjectSettings.class);
            
            //Configure the project, passing the model, extension, and the relevant project. Which does not need to be the root, but can be.
            toPerform.idea(project, model, ideaExt);
        });
    }
    
    /**
     * Applies the specified configuration action to configure eclipse IDE projects only.
     *
     * <p>This does not apply the eclipse IDEs' respective plugins, but will perform
     * actions when those plugins are applied.</p>
     *
     * @param toPerform the actions to perform
     */
    public void onEclipse(final EclipseIdeImportAction toPerform) {
        //When the Eclipse plugin is available, configure it
        project.getPlugins().withType(EclipsePlugin.class, plugin -> {
            //Do not configure the eclipse plugin if we are not importing.
            if (!isEclipseImport()) {
                return;
            }
            
            //Grab the eclipse model so we can extend it. -> Done on the root project so that the model is available to all subprojects.
            //And so that post sync tasks are only ran once for all subprojects.
            EclipseModel model = project.getExtensions().findByType(EclipseModel.class);
            if (model == null) {
                model = rootProject.getExtensions().findByType(EclipseModel.class);
                return;
            }
            
            //Configure the project, passing the model and the relevant project. Which does not need to be the root, but can be.
            toPerform.eclipse(project, model);
        });
    }
    
    /**
     * Applies the specified configuration action to configure gradle run projects only.
     *
     * @param toPerform the actions to perform
     */
    public void onGradle(final GradleIdeImportAction toPerform) {
        if (!isEclipseImport() && !isIdeaImport()) {
            toPerform.gradle(project);
        }
    }
    
    public interface GradleIdeImportAction {
        
        /**
         * Configure a gradle project.
         *
         * @param project the project being imported
         */
        default void gradle(Project project) {}
    }
    
    /**
     * A configuration action for idea IDE projects.
     */
    public interface IdeaIdeImportAction {
        
        /**
         * Configure an IntelliJ project.
         *
         * @param project       the project to configure on import
         * @param idea          the basic idea gradle extension
         * @param ideaExtension JetBrain's extensions to the base idea model
         */
        void idea(Project project, IdeaModel idea, ProjectSettings ideaExtension);
    }
    
    /**
     * A configuration action for eclipse IDE projects.
     */
    public interface EclipseIdeImportAction {
        
        /**
         * Configure an eclipse project.
         *
         * @param project the project being imported
         * @param eclipse the eclipse project model to modify
         */
        void eclipse(Project project, EclipseModel eclipse);
    }
    
    /**
     * A configuration action for IDE projects.
     */
    public interface IdeImportAction extends IdeaIdeImportAction, EclipseIdeImportAction, GradleIdeImportAction { }
}
